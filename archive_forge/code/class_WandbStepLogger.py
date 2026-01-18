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
class WandbStepLogger(WandbLoggerBase):

    def __init__(self, run: Run, step: int):
        super().__init__(run)
        self._step = step

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> MetricLogger:
        raise ValueError("can't create child logger")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.run.log(metrics, commit=True, step=self._step)

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError("can't log parameters from a step logger")

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        self.run.log(metadata, commit=True, step=self._step)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass