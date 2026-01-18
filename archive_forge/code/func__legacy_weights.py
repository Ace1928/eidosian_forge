import json
import os
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
def _legacy_weights(layer):
    """DO NOT USE.

  For legacy reason, the layer.weights was in the order of
  [self.trainable_weights + self.non_trainable_weights], and this order was
  used for preserving the weights in h5 format. The new order of layer.weights
  are the same as layer.get_weights() which is more intuitive for user. To
  keep supporting the existing saved h5 file, this method should be used to
  save/load weights. In future version, we will delete this method and
  introduce a breaking change for h5 and stay with the new order for weights.

  Args:
    layer: a `tf.keras.Model` or `tf.keras.layers.Layer` instance.

  Returns:
    A list of variables with the order of trainable_weights, followed by
      non_trainable_weights.
  """
    weights = layer.trainable_weights + layer.non_trainable_weights
    if any((not isinstance(w, variables_module.Variable) for w in weights)):
        raise NotImplementedError("Save or restore weights that is not an instance of `tf.Variable` is not supported in h5, use `save_format='tf'` instead. Got a model or layer {} with weights {}".format(layer.__class__.__name__, weights))
    return weights