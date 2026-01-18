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
def _log_keras_model(history, args):

    def _infer_model_signature(input_data_slice):
        original_stop_training = history.model.stop_training
        model_output = history.model.predict(input_data_slice)
        history.model.stop_training = original_stop_training
        return infer_signature(input_data_slice, model_output)
    from mlflow.tensorflow.autologging import extract_tf_keras_input_example

    def _get_tf_keras_input_example_slice():
        input_training_data = args[0]
        keras_input_example_slice = extract_tf_keras_input_example(input_training_data)
        if keras_input_example_slice is None:
            raise MlflowException(f'Cannot log input example or model signature for input with type {type(input_training_data)}. TensorFlow Keras autologging can only log input examples and model signatures for the following input types: numpy.ndarray, dict[string -> numpy.ndarray], tensorflow.keras.utils.Sequence, and tensorflow.data.Dataset (TensorFlow >= 2.1.0 required)', INVALID_PARAMETER_VALUE)
        return keras_input_example_slice
    input_example, signature = resolve_input_example_and_signature(_get_tf_keras_input_example_slice, _infer_model_signature, log_input_examples, log_model_signatures, _logger)
    log_model(model=history.model, artifact_path='model', input_example=input_example, signature=signature, registered_model_name=get_autologging_config(FLAVOR_NAME, 'registered_model_name', None), saved_model_kwargs=saved_model_kwargs, keras_model_kwargs=keras_model_kwargs)