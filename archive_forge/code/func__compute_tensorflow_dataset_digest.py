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
def _compute_tensorflow_dataset_digest(self, dataset, targets=None) -> str:
    """Computes a digest for the given Tensorflow dataset.

        Args:
            dataset: A Tensorflow dataset.

        Returns:
            A string digest.
        """
    import pandas as pd
    import tensorflow as tf
    hashable_elements = []

    def hash_tf_dataset_iterator_element(element):
        if element is None:
            return
        flat_element = tf.nest.flatten(element)
        flattened_array = np.concatenate([x.flatten() for x in flat_element])
        trimmed_array = flattened_array[0:MAX_ROWS]
        try:
            hashable_elements.append(pd.util.hash_array(trimmed_array))
        except TypeError:
            hashable_elements.append(np.int64(trimmed_array.size))
    for element in dataset.as_numpy_iterator():
        hash_tf_dataset_iterator_element(element)
    if targets is not None:
        for element in targets.as_numpy_iterator():
            hash_tf_dataset_iterator_element(element)
    return get_normalized_md5_digest(hashable_elements)