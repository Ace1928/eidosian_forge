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
def _convert_struct_values(result: Union[pandas.DataFrame, Dict[str, Any]], result_type):
    """
    Convert spark StructType values to spark dataframe column values.
    """
    from pyspark.sql.types import ArrayType, StructType
    if not isinstance(result_type, StructType):
        raise MlflowException.invalid_parameter_value(f'result_type must be StructType, got {result_type.simpleString()}')
    if not isinstance(result, (dict, pandas.DataFrame)):
        raise MlflowException.invalid_parameter_value(f'Unsupported result type {type(result)}, expected dict or pandas DataFrame')
    spark_primitive_type_to_np_type = _get_spark_primitive_type_to_np_type()
    is_pandas_df = isinstance(result, pandas.DataFrame)
    result_dict = {}
    for field_name in result_type.fieldNames():
        field_type = result_type[field_name].dataType
        field_values = result[field_name]
        if type(field_type) in spark_primitive_type_to_np_type:
            np_type = spark_primitive_type_to_np_type[type(field_type)]
            if is_pandas_df:
                field_values = field_values.astype(np_type)
            else:
                field_values = None if _is_none_or_nan(field_values) else np.array(field_values, dtype=np_type).item()
        elif isinstance(field_type, ArrayType):
            if is_pandas_df:
                field_values = pandas.Series((_convert_array_values(field_value, field_type) for field_value in field_values))
            else:
                field_values = _convert_array_values(field_values, field_type)
        elif isinstance(field_type, StructType):
            if is_pandas_df:
                field_values = pandas.Series([_convert_struct_values(field_value, field_type) for field_value in field_values])
            else:
                field_values = _convert_struct_values(field_values, field_type)
        else:
            raise MlflowException.invalid_parameter_value(f'Unsupported field type {field_type.simpleString()} in struct type.')
        result_dict[field_name] = field_values
    if is_pandas_df:
        return pandas.DataFrame(result_dict)
    return result_dict