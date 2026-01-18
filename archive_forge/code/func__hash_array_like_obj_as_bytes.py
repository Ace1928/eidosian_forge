import json
import keyword
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name, insecure_hash
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string
def _hash_array_like_obj_as_bytes(data):
    """
    Helper method to convert pandas dataframe/numpy array/list into bytes for
    MD5 calculation purpose.
    """
    if isinstance(data, pd.DataFrame):
        if 'pyspark' in sys.modules:
            from pyspark.ml.linalg import Vector as spark_vector_type
        else:
            spark_vector_type = None

        def _hash_array_like_element_as_bytes(v):
            if spark_vector_type is not None:
                if isinstance(v, spark_vector_type):
                    return _hash_ndarray_as_bytes(v.toArray())
            if isinstance(v, (dict, list, np.ndarray)):
                return _hash_data_as_bytes(v)
            return v
        data = data.applymap(_hash_array_like_element_as_bytes)
        return _hash_uint64_ndarray_as_bytes(pd.util.hash_pandas_object(data))
    elif isinstance(data, np.ndarray) and len(data) > 0 and isinstance(data[0], list):
        hashable = np.array((str(val) for val in data))
        return _hash_ndarray_as_bytes(hashable)
    elif isinstance(data, np.ndarray) and len(data) > 0 and isinstance(data[0], np.ndarray):
        hashable = np.array(data.tolist())
        return _hash_ndarray_as_bytes(hashable)
    elif isinstance(data, np.ndarray):
        return _hash_ndarray_as_bytes(data)
    elif isinstance(data, list):
        return _hash_ndarray_as_bytes(np.array(data))
    else:
        raise ValueError('Unsupported data type.')