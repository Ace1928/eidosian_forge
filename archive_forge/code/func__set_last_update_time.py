from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import ExperimentTag as ProtoExperimentTag
def _set_last_update_time(self, last_update_time):
    self._last_update_time = last_update_time