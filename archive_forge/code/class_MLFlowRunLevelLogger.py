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
class MLFlowRunLevelLogger(MLFlowExperimentLevelLogger):

    def __init__(self, parent: MLFlowExperimentLevelLogger, name: Optional[str]=None, description: Optional[str]=None, run_id: Optional[str]=None, tags: Optional[Dict[str, Any]]=None):
        super().__init__(parent.client, parent.experiment)
        if run_id is None:
            t = {} if tags is None else dict(tags)
            if name is None:
                name = self.unique_id
            t[MLFLOW_RUN_NAME] = name
            if description is not None:
                t[MLFLOW_RUN_NOTE] = description
            if isinstance(parent, MLFlowRunLevelLogger):
                t[MLFLOW_PARENT_RUN_ID] = parent.run_id
                t['parent'] = parent.run_name if parent.run_name is not None else parent.run_id[-5:]
                self._is_child = True
            else:
                self._is_child = False
            resolved_tags = context_registry.resolve_tags(t)
            self._run = self.client.create_run(self.experiment_id, tags=resolved_tags)
        else:
            self._run = self.client.get_run(run_id)
            self._is_child = False
        self._step = 0
        os.environ['MLFLOW_RUN_ID'] = self.run_id

    def __getstate__(self) -> Dict[str, Any]:
        return dict(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri, experiment_id=self.experiment_id, run_id=self.run_id, is_child=self._is_child, step=self._step, unique_id=self.unique_id)

    def __setstate__(self, data: Dict[str, Any]) -> None:
        self._unique_id = data['unique_id']
        self._client = MlflowClient(tracking_uri=data['tracking_uri'], registry_uri=data['registry_uri'])
        mlflow.tracking.set_tracking_uri(data['tracking_uri'])
        mlflow.tracking.set_registry_uri(data['registry_uri'])
        self._experiment = self._client.get_experiment(data['experiment_id'])
        self._run = self._client.get_run(data['run_id'])
        self._is_child = data['is_child']
        self._step = data['step']

    @property
    def run(self) -> Run:
        return self._run

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    @property
    def run_name(self) -> Optional[str]:
        return self.run.data.tags.get(MLFLOW_RUN_NAME, None)

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> 'MetricLogger':
        if is_step:
            self._step += 1
            return MLFlowStepLevelLogger(self, self._step - 1)
        else:
            assert not self._is_child
            return MLFlowRunLevelLogger(self, name=name, description=description)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.set_terminated(self.run_id, 'FINISHED' if exc_type is None else 'FAILED')