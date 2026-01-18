import json
import logging
import math
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
from google.protobuf.json_format import ParseDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from wandb import Artifact
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_settings_pb2
from wandb.proto import wandb_telemetry_pb2 as telem_pb
from wandb.sdk.interface.interface import file_policy_to_enum
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context
from wandb.sdk.internal.sender import SendManager
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.util import coalesce, recursive_cast_dictlike_to_dict
from .protocols import ImporterRun
@dataclass
class RecordMaker:
    run: ImporterRun
    interface: InterfaceQueue = InterfaceQueue()

    @property
    def run_dir(self) -> str:
        p = Path(f'{ROOT_DIR}/{self.run.run_id()}/wandb')
        p.mkdir(parents=True, exist_ok=True)
        return f'{ROOT_DIR}/{self.run.run_id()}'

    def make_artifacts_only_records(self, artifacts: Optional[Iterable[Artifact]]=None, used_artifacts: Optional[Iterable[Artifact]]=None) -> Iterable[pb.Record]:
        """Only make records required to upload artifacts.

        Escape hatch for adding extra artifacts to a run.
        """
        yield self._make_run_record()
        if used_artifacts:
            for art in used_artifacts:
                yield self._make_artifact_record(art, use_artifact=True)
        if artifacts:
            for art in artifacts:
                yield self._make_artifact_record(art)

    def make_records(self, config: SendManagerConfig) -> Iterable[pb.Record]:
        """Make all the records that constitute a run."""
        yield self._make_run_record()
        yield self._make_telem_record()
        include_artifacts = config.log_artifacts or config.use_artifacts
        yield self._make_files_record(include_artifacts, config.files, config.media, config.code)
        if config.use_artifacts:
            if (used_artifacts := self.run.used_artifacts()) is not None:
                for artifact in used_artifacts:
                    yield self._make_artifact_record(artifact, use_artifact=True)
        if config.log_artifacts:
            if (artifacts := self.run.artifacts()) is not None:
                for artifact in artifacts:
                    yield self._make_artifact_record(artifact)
        if config.history:
            yield from self._make_history_records()
        if config.summary:
            yield self._make_summary_record()
        if config.terminal_output:
            if (lines := self.run.logs()) is not None:
                for line in lines:
                    yield self._make_output_record(line)

    def _make_run_record(self) -> pb.Record:
        run = pb.RunRecord()
        run.run_id = self.run.run_id()
        run.entity = self.run.entity()
        run.project = self.run.project()
        run.display_name = coalesce(self.run.display_name())
        run.notes = coalesce(self.run.notes(), '')
        run.tags.extend(coalesce(self.run.tags(), []))
        run.start_time.FromMilliseconds(self.run.start_time())
        host = self.run.host()
        if host is not None:
            run.host = host
        runtime = self.run.runtime()
        if runtime is not None:
            run.runtime = runtime
        run_group = self.run.run_group()
        if run_group is not None:
            run.run_group = run_group
        config = self.run.config()
        if '_wandb' not in config:
            config['_wandb'] = {}
        config['_wandb']['code_path'] = self.run.code_path()
        config['_wandb']['python_version'] = self.run.python_version()
        config['_wandb']['cli_version'] = self.run.cli_version()
        self.interface._make_config(data=config, obj=run.config)
        return self.interface._make_record(run=run)

    def _make_output_record(self, line) -> pb.Record:
        output_raw = pb.OutputRawRecord()
        output_raw.output_type = pb.OutputRawRecord.OutputType.STDOUT
        output_raw.line = line
        return self.interface._make_record(output_raw=output_raw)

    def _make_summary_record(self) -> pb.Record:
        d: dict = {**self.run.summary(), '_runtime': self.run.runtime()}
        d = recursive_cast_dictlike_to_dict(d)
        summary = self.interface._make_summary_from_dict(d)
        return self.interface._make_record(summary=summary)

    def _make_history_records(self) -> Iterable[pb.Record]:
        for metrics in self.run.metrics():
            history = pb.HistoryRecord()
            for k, v in metrics.items():
                item = history.item.add()
                item.key = k
                if isinstance(v, float) and math.isnan(v) or v == 'NaN':
                    v = np.NaN
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                else:
                    v = json.dumps(v)
                item.value_json = v
            rec = self.interface._make_record(history=history)
            yield rec

    def _make_files_record(self, artifacts: bool, files: bool, media: bool, code: bool) -> pb.Record:
        run_files = self.run.files()
        metadata_fname = f'{self.run_dir}/files/wandb-metadata.json'
        if not files or run_files is None:
            metadata_fname = self._make_metadata_file()
            run_files = [(metadata_fname, 'end')]
        files_record = pb.FilesRecord()
        for path, policy in run_files:
            if not artifacts and path.startswith('artifact/'):
                continue
            if not media and path.startswith('media/'):
                continue
            if not code and path.startswith('code/'):
                continue
            if 'media' in path:
                p = Path(path)
                path = str(p.relative_to(f'{self.run_dir}/files'))
            f = files_record.files.add()
            f.path = path
            f.policy = file_policy_to_enum(policy)
        return self.interface._make_record(files=files_record)

    def _make_artifact_record(self, artifact: Artifact, use_artifact=False) -> pb.Record:
        proto = self.interface._make_artifact(artifact)
        proto.run_id = str(self.run.run_id())
        proto.project = str(self.run.project())
        proto.entity = str(self.run.entity())
        proto.user_created = use_artifact
        proto.use_after_commit = use_artifact
        proto.finalize = True
        aliases = artifact._aliases
        aliases += ['latest', 'imported']
        for alias in aliases:
            proto.aliases.append(alias)
        return self.interface._make_record(artifact=proto)

    def _make_telem_record(self) -> pb.Record:
        telem = telem_pb.TelemetryRecord()
        feature = telem_pb.Feature()
        feature.importer_mlflow = True
        telem.feature.CopyFrom(feature)
        cli_version = self.run.cli_version()
        if cli_version:
            telem.cli_version = cli_version
        python_version = self.run.python_version()
        if python_version:
            telem.python_version = python_version
        return self.interface._make_record(telemetry=telem)

    def _make_metadata_file(self) -> str:
        missing_text = 'This data was not captured'
        files_dir = f'{self.run_dir}/files'
        os.makedirs(files_dir, exist_ok=True)
        d = {}
        d['os'] = coalesce(self.run.os_version(), missing_text)
        d['python'] = coalesce(self.run.python_version(), missing_text)
        d['program'] = coalesce(self.run.program(), missing_text)
        d['cuda'] = coalesce(self.run.cuda_version(), missing_text)
        d['host'] = coalesce(self.run.host(), missing_text)
        d['username'] = coalesce(self.run.username(), missing_text)
        d['executable'] = coalesce(self.run.executable(), missing_text)
        gpus_used = self.run.gpus_used()
        if gpus_used is not None:
            d['gpu_devices'] = json.dumps(gpus_used)
            d['gpu_count'] = json.dumps(len(gpus_used))
        cpus_used = self.run.cpus_used()
        if cpus_used is not None:
            d['cpu_count'] = json.dumps(self.run.cpus_used())
        mem_used = self.run.memory_used()
        if mem_used is not None:
            d['memory'] = json.dumps({'total': self.run.memory_used()})
        fname = f'{files_dir}/wandb-metadata.json'
        with open(fname, 'w') as f:
            f.write(json.dumps(d))
        return fname