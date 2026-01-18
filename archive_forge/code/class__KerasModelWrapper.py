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
class _KerasModelWrapper:

    def __init__(self, keras_model, signature):
        self.keras_model = keras_model
        self.signature = signature

    def predict(self, data, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                                        release without warning.

        Returns
            Model predictions.
        """
        if isinstance(data, pandas.DataFrame):
            return pandas.DataFrame(self.keras_model.predict(data.values), index=data.index)
        supported_input_types = (np.ndarray, list, tuple, dict)
        if not isinstance(data, supported_input_types):
            raise MlflowException(f'Unsupported input data type: {type(data)}. Must be one of: {[x.__name__ for x in supported_input_types]}', INVALID_PARAMETER_VALUE)
        return self.keras_model.predict(data)