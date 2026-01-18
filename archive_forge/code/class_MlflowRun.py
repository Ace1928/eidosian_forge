import itertools
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import mlflow
from packaging.version import Version  # type: ignore
import wandb
from wandb import Artifact
from .internals import internal
from .internals.util import Namespace, for_each
class MlflowRun:

    def __init__(self, run, mlflow_client):
        self.run = run
        self.mlflow_client: mlflow.MlflowClient = mlflow_client

    def run_id(self) -> str:
        return self.run.info.run_id

    def entity(self) -> str:
        return self.run.info.user_id

    def project(self) -> str:
        return 'imported-from-mlflow'

    def config(self) -> Dict[str, Any]:
        conf = self.run.data.params
        tags = {k: v for k, v in self.run.data.tags.items() if not k.startswith('mlflow.')}
        return {**conf, 'imported_mlflow_tags': tags}

    def summary(self) -> Dict[str, float]:
        return self.run.data.metrics

    def metrics(self) -> Iterable[Dict[str, float]]:
        d: Dict[int, Dict[str, float]] = defaultdict(dict)
        for k in self.run.data.metrics.keys():
            metric = self.mlflow_client.get_metric_history(self.run.info.run_id, k)
            for item in metric:
                d[item.step][item.key] = item.value
        for k, v in d.items():
            yield {'_step': k, **v}

    def run_group(self) -> Optional[str]:
        return f'Experiment {self.run.info.experiment_id}'

    def job_type(self) -> Optional[str]:
        return f'User {self.run.info.user_id}'

    def display_name(self) -> str:
        if mlflow_version < Version('1.30.0'):
            return self.run.data.tags['mlflow.runName']
        return self.run.info.run_name

    def notes(self) -> Optional[str]:
        return self.run.data.tags.get('mlflow.note.content')

    def tags(self) -> Optional[List[str]]:
        ...

    def artifacts(self) -> Optional[Iterable[Artifact]]:
        if mlflow_version < Version('2.0.0'):
            dir_path = self.mlflow_client.download_artifacts(run_id=self.run.info.run_id, path='')
        else:
            dir_path = mlflow.artifacts.download_artifacts(run_id=self.run.info.run_id)
        artifact_name = self._handle_incompatible_strings(self.display_name())
        art = wandb.Artifact(artifact_name, 'imported-artifacts')
        art.add_dir(dir_path)
        return [art]

    def used_artifacts(self) -> Optional[Iterable[Artifact]]:
        ...

    def os_version(self) -> Optional[str]:
        ...

    def python_version(self) -> Optional[str]:
        ...

    def cuda_version(self) -> Optional[str]:
        ...

    def program(self) -> Optional[str]:
        ...

    def host(self) -> Optional[str]:
        ...

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
        end_time = self.run.info.end_time // 1000 if self.run.info.end_time is not None else self.start_time()
        return end_time - self.start_time()

    def start_time(self) -> Optional[int]:
        return self.run.info.start_time // 1000

    def code_path(self) -> Optional[str]:
        ...

    def cli_version(self) -> Optional[str]:
        ...

    def files(self) -> Optional[Iterable[Tuple[str, str]]]:
        ...

    def logs(self) -> Optional[Iterable[str]]:
        ...

    @staticmethod
    def _handle_incompatible_strings(s: str) -> str:
        valid_chars = '[^a-zA-Z0-9_\\-\\.]'
        replacement = '__'
        return re.sub(valid_chars, replacement, s)