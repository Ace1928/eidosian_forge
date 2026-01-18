import os
from typing import Any, Dict, Optional, Union
import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import (
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
class MLFlowStepLevelLogger(MLFlowExperimentLevelLogger):

    def __init__(self, parent: MLFlowRunLevelLogger, step: int):
        super().__init__(parent.client, parent.experiment)
        self._run = parent.run
        self._step = step
        os.environ['MLFLOW_RUN_ID'] = self.run_id

    @property
    def run(self) -> Run:
        return self._run

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> 'MetricLogger':
        raise TuneRuntimeError(str(type(self)) + " can't have a child logger")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, k, v, step=self._step)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass