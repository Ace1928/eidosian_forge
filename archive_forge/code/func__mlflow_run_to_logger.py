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
@parse_logger.candidate(lambda obj: isinstance(obj, (Run, ActiveRun)))
def _mlflow_run_to_logger(obj: Union[Run, ActiveRun]) -> 'MLFlowRunLevelLogger':
    if MLFLOW_PARENT_RUN_ID in obj.data.tags:
        pr = mlflow.get_run(obj.data.tags[MLFLOW_PARENT_RUN_ID])
        parent: Any = _mlflow_run_to_logger(pr)
    else:
        client = MlflowClient()
        parent = MLFlowExperimentLevelLogger(client, mlflow.get_experiment(obj.info.experiment_id))
    return MLFlowRunLevelLogger(parent, run_id=obj.info.run_id)