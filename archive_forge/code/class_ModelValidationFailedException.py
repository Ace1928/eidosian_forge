from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import deprecated
class ModelValidationFailedException(MlflowException):

    def __init__(self, message, **kwargs):
        super().__init__(message, error_code=BAD_REQUEST, **kwargs)