from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated
class MetricThresholdClassException(MlflowException):

    def __init__(self, _message, **kwargs):
        message = 'Could not instantiate MetricThreshold class: ' + _message
        super().__init__(message, error_code=INVALID_PARAMETER_VALUE, **kwargs)