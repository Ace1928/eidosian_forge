import itertools
import json
import logging
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import patch
import filelock
import polars as pl
import requests
import urllib3
import yaml
from wandb_gql import gql
import wandb
import wandb.apis.reports as wr
from wandb.apis.public import ArtifactCollection, Run
from wandb.apis.public.files import File
from wandb.apis.reports import Report
from wandb.util import coalesce, remove_keys_with_none_values
from . import validation
from .internals import internal
from .internals.protocols import PathStr, Policy
from .internals.util import Namespace, for_each
class WandbRun:

    def __init__(self, run: Run, *, src_base_url: str, src_api_key: str, dst_base_url: str, dst_api_key: str) -> None:
        self.run = run
        self.api = wandb.Api(api_key=src_api_key, overrides={'base_url': src_base_url})
        self.dst_api = wandb.Api(api_key=dst_api_key, overrides={'base_url': dst_base_url})
        self._files: Optional[Iterable[Tuple[str, str]]] = None
        self._artifacts: Optional[Iterable[Artifact]] = None
        self._used_artifacts: Optional[Iterable[Artifact]] = None
        self._parquet_history_paths: Optional[Iterable[str]] = None

    def __repr__(self) -> str:
        s = os.path.join(self.entity(), self.project(), self.run_id())
        return f'WandbRun({s})'

    def run_id(self) -> str:
        return self.run.id

    def entity(self) -> str:
        return self.run.entity

    def project(self) -> str:
        return self.run.project

    def config(self) -> Dict[str, Any]:
        return self.run.config

    def summary(self) -> Dict[str, float]:
        s = self.run.summary
        return s

    def metrics(self) -> Iterable[Dict[str, float]]:
        if self._parquet_history_paths is None:
            self._parquet_history_paths = list(self._get_parquet_history_paths())
        if self._parquet_history_paths:
            rows = self._get_rows_from_parquet_history_paths()
        else:
            logger.warn('No parquet files detected; using scan history (this may not be reliable)')
            rows = self.run.scan_history()
        for row in rows:
            row = remove_keys_with_none_values(row)
            yield row

    def run_group(self) -> Optional[str]:
        return self.run.group

    def job_type(self) -> Optional[str]:
        return self.run.job_type

    def display_name(self) -> str:
        return self.run.display_name

    def notes(self) -> Optional[str]:
        previous_link = f'Imported from: {self.run.url}'
        previous_author = f'Author: {self.run.user.username}'
        header = [previous_link, previous_author]
        previous_notes = self.run.notes or ''
        return '\n'.join(header) + '\n---\n' + previous_notes

    def tags(self) -> Optional[List[str]]:
        return self.run.tags

    def artifacts(self) -> Optional[Iterable[Artifact]]:
        if self._artifacts is None:
            _artifacts = []
            for art in self.run.logged_artifacts():
                a = _clone_art(art)
                _artifacts.append(a)
            self._artifacts = _artifacts
        yield from self._artifacts

    def used_artifacts(self) -> Optional[Iterable[Artifact]]:
        if self._used_artifacts is None:
            _used_artifacts = []
            for art in self.run.used_artifacts():
                a = _clone_art(art)
                _used_artifacts.append(a)
            self._used_artifacts = _used_artifacts
        yield from self._used_artifacts

    def os_version(self) -> Optional[str]:
        ...

    def python_version(self) -> Optional[str]:
        return self._metadata_file().get('python')

    def cuda_version(self) -> Optional[str]:
        ...

    def program(self) -> Optional[str]:
        ...

    def host(self) -> Optional[str]:
        return self._metadata_file().get('host')

    def username(self) -> Optional[str]:
        ...

    def executable(self) -> Optional[str]:
        ...

    def gpus_used(self) -> Optional[str]:
        ...

    def cpus_used(self) -> Optional[int]:
        ...

    def memory_used(self) -> Optional[int]:
        ...

    def runtime(self) -> Optional[int]:
        wandb_runtime = self.run.summary.get('_wandb', {}).get('runtime')
        base_runtime = self.run.summary.get('_runtime')
        if (t := coalesce(wandb_runtime, base_runtime)) is None:
            return t
        return int(t)

    def start_time(self) -> Optional[int]:
        t = dt.fromisoformat(self.run.created_at).timestamp() * 1000
        return int(t)

    def code_path(self) -> Optional[str]:
        path = self._metadata_file().get('codePath', '')
        return f'code/{path}'

    def cli_version(self) -> Optional[str]:
        return self._config_file().get('_wandb', {}).get('value', {}).get('cli_version')

    def files(self) -> Optional[Iterable[Tuple[PathStr, Policy]]]:
        if self._files is None:
            files_dir = f'{internal.ROOT_DIR}/{self.run_id()}/files'
            _files = []
            for f in self.run.files():
                f: File
                if f.size == 0:
                    continue
                if 'wandb_manifest.json.deadlist' in f.name:
                    continue
                result = f.download(files_dir, exist_ok=True, api=self.api)
                file_and_policy = (result.name, 'end')
                _files.append(file_and_policy)
            self._files = _files
        yield from self._files

    def logs(self) -> Optional[Iterable[str]]:
        if (fname := self._find_in_files('output.log')) is None:
            return
        with open(fname) as f:
            yield from f.readlines()

    def _metadata_file(self) -> Dict[str, Any]:
        if (fname := self._find_in_files('wandb-metadata.json')) is None:
            return {}
        with open(fname) as f:
            return json.loads(f.read())

    def _config_file(self) -> Dict[str, Any]:
        if (fname := self._find_in_files('config.yaml')) is None:
            return {}
        with open(fname) as f:
            return yaml.safe_load(f) or {}

    def _get_rows_from_parquet_history_paths(self) -> Iterable[Dict[str, Any]]:
        if not (paths := self._get_parquet_history_paths()):
            yield {}
            return
        dfs = [pl.read_parquet(p) for path in paths for p in Path(path).glob('*.parquet')]
        if '_step' in (df := _merge_dfs(dfs)):
            df = df.with_columns(pl.col('_step').cast(pl.Int64))
        yield from df.iter_rows(named=True)

    def _get_parquet_history_paths(self) -> Iterable[str]:
        if self._parquet_history_paths is None:
            paths = []
            for art in self.run.logged_artifacts():
                if art.type != 'wandb-history':
                    continue
                if (path := _download_art(art, root=f'{SRC_ART_PATH}/{art.name}')) is None:
                    continue
                paths.append(path)
            self._parquet_history_paths = paths
        yield from self._parquet_history_paths

    def _find_in_files(self, name: str) -> Optional[str]:
        if (files := self.files()):
            for path, _ in files:
                if name in path:
                    return path
        return None