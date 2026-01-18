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
def _parse_dataset(*keras_fit_args, **keras_fit_kwargs):
    """Parse dataset from `keras.Model.fit` args and kwargs."""
    x = keras_fit_kwargs['x'] if 'x' in keras_fit_kwargs else keras_fit_args[0]
    if 'y' in keras_fit_kwargs:
        y = keras_fit_kwargs['y']
    elif len(keras_fit_args) >= 2:
        y = keras_fit_args[1]
    else:
        y = None
    return (x, y)