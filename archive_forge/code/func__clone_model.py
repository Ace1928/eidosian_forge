import contextlib
import copy
from io import BytesIO
from typing import Any, Dict, List, Optional
import catalogue
import numpy
from ..backends import Ops, get_current_ops
from ..compat import cupy, h5py
from ..compat import tensorflow as tf
from ..optimizers import Optimizer
from ..types import ArgsKwargs, ArrayXd
from ..util import get_array_module
from .shim import Shim
def _clone_model(self):
    """similar to tf.keras.models.clone_model()
        But the tf.keras.models.clone_model changes the names of tf.Variables.
        This method even preserves that
        """
    model_json_config = self._model.to_json()
    tf.keras.backend.clear_session()
    self._model = tf.keras.models.model_from_json(model_json_config)
    self._load_weights_from_state_dict()