import json
import logging
from mlflow.protos.databricks_pb2 import (
class InvalidUrlException(MlflowException):
    """Exception thrown when a http request fails to send due to an invalid URL"""
    pass