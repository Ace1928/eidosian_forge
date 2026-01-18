import logging
import sys
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from wandb.sdk.artifacts.artifact import Artifact
@runtime_checkable
class ImporterRun(Protocol):

    def run_id(self) -> str:
        ...

    def entity(self) -> str:
        ...

    def project(self) -> str:
        ...

    def config(self) -> Dict[str, Any]:
        ...

    def summary(self) -> Dict[str, float]:
        ...

    def metrics(self) -> Iterable[Dict[str, float]]:
        """Metrics for the run.

        We expect metrics in this shape:

        [
            {'metric1': 1, 'metric2': 1, '_step': 0},
            {'metric1': 2, 'metric2': 4, '_step': 1},
            {'metric1': 3, 'metric2': 9, '_step': 2},
            ...
        ]

        You can also submit metrics in this shape:
        [
            {'metric1': 1, '_step': 0},
            {'metric2': 1, '_step': 0},
            {'metric1': 2, '_step': 1},
            {'metric2': 4, '_step': 1},
            ...
        ]
        """
        ...

    def run_group(self) -> Optional[str]:
        ...

    def job_type(self) -> Optional[str]:
        ...

    def display_name(self) -> str:
        ...

    def notes(self) -> Optional[str]:
        ...

    def tags(self) -> Optional[List[str]]:
        ...

    def artifacts(self) -> Optional[Iterable[Artifact]]:
        ...

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
        ...

    def start_time(self) -> Optional[int]:
        ...

    def code_path(self) -> Optional[str]:
        ...

    def cli_version(self) -> Optional[str]:
        ...

    def files(self) -> Optional[Iterable[Tuple[PathStr, Policy]]]:
        ...

    def logs(self) -> Optional[Iterable[str]]:
        ...