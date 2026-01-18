import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from packaging.version import Version
import mlflow
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags
def _convert_data_to_dict(data, key):
    if isinstance(data, dict):
        return {f'{key}-{k}': v for k, v in data.items()}
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, str):
        return {key: [data]}
    raise MlflowException('Unsupported data type.')