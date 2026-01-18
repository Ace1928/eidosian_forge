import logging
import os
import pathlib
import sys
import traceback
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.utils import reraise
from mlflow.utils.annotations import deprecated, keyword_only
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
class MLeapSerializationException(MlflowException):
    """Exception thrown when a model or DataFrame cannot be serialized in MLeap format."""