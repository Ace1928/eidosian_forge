import os
from typing import Any, Dict, Optional
import wandb
from triad import assert_or_throw
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
from wandb.env import get_project
from wandb.sdk.lib.apikey import api_key
from wandb.wandb_run import Run
class WandbLoggerBase(MetricLogger):

    def __init__(self, run: Optional[Run]=None):
        super().__init__()
        self._run = run

    def __getstate__(self) -> Dict[str, Any]:
        raise TuneRuntimeError(str(type(self)) + ' is not serializable')

    @property
    def run(self) -> Run:
        assert_or_throw(self._run is not None, NotImplementedError('wandb run is not available'))
        return self._run

    @property
    def api_key(self) -> str:
        return api_key(settings=None if self._run is None else self.run._settings)

    @property
    def project_name(self) -> str:
        return self.run.project_name()

    @property
    def group(self) -> str:
        return self.run.group

    @property
    def run_id(self) -> str:
        return self.run.id

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.run.log(metrics, commit=True)

    def log_params(self, params: Dict[str, Any]) -> None:
        self.run.config.update(params)

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        self.run.log(metadata, commit=True)

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> MetricLogger:
        raise TuneRuntimeError(str(type(self)) + " can't have a child logger")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        exit_code = 0 if exc_type is None else 1
        if self._run is not None:
            self.run.finish(exit_code)