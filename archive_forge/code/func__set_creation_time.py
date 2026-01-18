from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import ExperimentTag as ProtoExperimentTag
def _set_creation_time(self, creation_time):
    self._creation_time = creation_time