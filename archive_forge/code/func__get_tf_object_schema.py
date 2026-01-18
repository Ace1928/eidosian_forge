import json
import logging
from functools import cached_property
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.digest_utils import (
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema
@staticmethod
def _get_tf_object_schema(tf_object) -> Schema:
    import tensorflow as tf
    if isinstance(tf_object, tf.data.Dataset):
        numpy_data = next(tf_object.as_numpy_iterator())
        if isinstance(numpy_data, np.ndarray):
            return _infer_schema(numpy_data)
        elif isinstance(numpy_data, dict):
            return TensorFlowDataset._get_schema_from_tf_dataset_dict_numpy_data(numpy_data)
        elif isinstance(numpy_data, tuple):
            return TensorFlowDataset._get_schema_from_tf_dataset_tuple_numpy_data(numpy_data)
        else:
            raise MlflowException(f"Failed to infer schema for tf.data.Dataset due to unrecognized numpy iterator data type. Numpy iterator data types 'np.ndarray', 'dict', and 'tuple' are supported. Found: {type(numpy_data)}.", INVALID_PARAMETER_VALUE)
    elif tf.is_tensor(tf_object):
        return _infer_schema(tf_object.numpy())
    else:
        raise MlflowException(f'Cannot infer schema of an object that is not an instance of tf.data.Dataset or a TensorFlow Tensor. Found: {type(tf_object)}', INTERNAL_ERROR)