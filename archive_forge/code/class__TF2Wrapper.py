import importlib
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, NamedTuple, Optional
import numpy as np
import pandas
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.tensorflow_dataset import from_tensorflow
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tensorflow.callback import MlflowCallback, MlflowModelCheckpointCallback  # noqa: F401
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.types.schema import TensorSpec
from mlflow.utils import is_iterator
from mlflow.utils.autologging_utils import (
from mlflow.utils.checkpoint_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _TF2Wrapper:
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    ``predict(data: pandas.DataFrame) -> pandas.DataFrame``. For TensorFlow versions >= 2.0.0.
    """

    def __init__(self, model, infer):
        """
        Args:
            model: A Tensorflow SavedModel.
            infer: Tensorflow function returned by a saved model that is used for inference.
        """
        self.model = model
        self.infer = infer

    def predict(self, data, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                                        release without warning.

        Returns:
            Model predictions.
        """
        import tensorflow as tf
        feed_dict = {}
        if isinstance(data, dict):
            feed_dict = {k: tf.constant(v) for k, v in data.items()}
        elif isinstance(data, pandas.DataFrame):
            for df_col_name in list(data):
                val = data[df_col_name]
                val = val.values if isinstance(val, pandas.DataFrame) else np.array(val.to_list())
                feed_dict[df_col_name] = tf.constant(val)
        else:
            raise TypeError('Only dict and DataFrame input types are supported')
        raw_preds = self.infer(**feed_dict)
        pred_dict = {col_name: raw_preds[col_name].numpy() for col_name in raw_preds.keys()}
        for col in pred_dict.keys():
            if len(pred_dict[col].shape) != 1 and all((len(element) == 1 for element in pred_dict[col])):
                pred_dict[col] = pred_dict[col].ravel()
            else:
                pred_dict[col] = pred_dict[col].tolist()
        if isinstance(data, dict):
            return pred_dict
        else:
            return pandas.DataFrame.from_dict(data=pred_dict)