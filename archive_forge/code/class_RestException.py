import json
import logging
from mlflow.protos.databricks_pb2 import (
class RestException(MlflowException):
    """Exception thrown on non 200-level responses from the REST API"""

    def __init__(self, json):
        self.json = json
        error_code = json.get('error_code', ErrorCode.Name(INTERNAL_ERROR))
        message = '{}: {}'.format(error_code, json['message'] if 'message' in json else 'Response: ' + str(json))
        try:
            super().__init__(message, error_code=ErrorCode.Value(error_code))
        except ValueError:
            try:
                error_code = HTTP_STATUS_TO_ERROR_CODE[int(error_code)]
                super().__init__(message, error_code=ErrorCode.Value(error_code))
            except ValueError or KeyError:
                _logger.warning(f'Received error code not recognized by MLflow: {error_code}, this may indicate your request encountered an error before reaching MLflow server, e.g., within a proxy server or authentication / authorization service.')
                super().__init__(message)

    def __reduce__(self):
        """
        Overriding `__reduce__` to make `RestException` instance pickle-able.
        """
        return (RestException, (self.json,))