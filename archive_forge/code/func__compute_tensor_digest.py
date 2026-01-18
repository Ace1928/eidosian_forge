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
def _compute_tensor_digest(self, tensor_data, tensor_targets) -> str:
    """Computes a digest for the given Tensorflow tensor.

        Args:
            tensor_data: A Tensorflow tensor, representing the features.
            tensor_targets: A Tensorflow tensor, representing the targets. Optional.

        Returns:
            A string digest.
        """
    if tensor_targets is None:
        return compute_numpy_digest(tensor_data.numpy())
    else:
        return compute_numpy_digest(tensor_data.numpy(), tensor_targets.numpy())