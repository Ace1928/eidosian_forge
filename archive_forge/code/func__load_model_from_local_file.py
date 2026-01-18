import functools
import inspect
import logging
import os
import pickle
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _inspect_original_var_name, gorilla
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _load_model_from_local_file(path, serialization_format):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system.

    Args:
        path: Local filesystem path to the MLflow Model saved with the ``sklearn`` flavor
        serialization_format: The format in which the model was serialized. This should be one of
            the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
            ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(message=f'Unrecognized serialization format: {serialization_format}. Please specify one of the following supported formats: {SUPPORTED_SERIALIZATION_FORMATS}.', error_code=INVALID_PARAMETER_VALUE)
    with open(path, 'rb') as f:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(f)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle
            return cloudpickle.load(f)