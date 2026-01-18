import logging
import keras
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.tensorflow_dataset import from_tensorflow
from mlflow.exceptions import MlflowException
from mlflow.keras.callback import MlflowCallback
from mlflow.keras.save import log_model
from mlflow.keras.utils import get_model_signature
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import is_iterator
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
def _check_existing_mlflow_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, MlflowCallback):
            raise MlflowException('MLflow autologging must be turned off if an `MlflowCallback` is explicitly added to the callback list. You are creating an `MlflowCallback` while having autologging enabled. Please either call `mlflow.keras.autolog(disable=True)` to disable autologging or remove `MlflowCallback` from the callback list. ')