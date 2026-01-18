import collections
import functools
import importlib
import inspect
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import numpy as np
import pandas
import yaml
import mlflow
import mlflow.pyfunc.loaders
import mlflow.pyfunc.model
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import _DATABRICKS_FS_LOADER_MODULE, MLMODEL_FILE_NAME
from mlflow.models.signature import (
from mlflow.models.utils import (
from mlflow.protos.databricks_pb2 import (
from mlflow.pyfunc.model import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
from mlflow.utils import (
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils._spark_utils import modified_environ
from mlflow.utils.annotations import deprecated, developer_stable, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import (
def _cast_output_spec_to_spark_type(spec):
    from pyspark.sql.types import ArrayType
    from mlflow.types.schema import ColSpec, DataType, TensorSpec
    if isinstance(spec, ColSpec):
        return _convert_spec_type_to_spark_type(spec.type)
    elif isinstance(spec, TensorSpec):
        data_type = DataType.from_numpy_type(spec.type)
        if data_type is None:
            raise MlflowException(f'Model output tensor spec type {spec.type} is not supported in spark_udf.', error_code=INVALID_PARAMETER_VALUE)
        if len(spec.shape) == 1:
            return ArrayType(data_type.to_spark())
        elif len(spec.shape) == 2:
            return ArrayType(ArrayType(data_type.to_spark()))
        else:
            raise MlflowException(f"Only 1D or 2D tensors are supported as spark_udf return value, but model output '{spec.name}' has shape {spec.shape}.", error_code=INVALID_PARAMETER_VALUE)
    else:
        raise MlflowException(f'Unknown schema output spec {spec}.', error_code=INVALID_PARAMETER_VALUE)