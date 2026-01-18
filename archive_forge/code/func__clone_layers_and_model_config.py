from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _clone_layers_and_model_config(model, input_layers, layer_fn):
    """Clones all layers, and returns the model config without serializing layers.

  This function ensures that only the node graph is retrieved when getting the
  model config. The `layer_fn` used to clone layers might not rely on
  `layer.get_config()`, so some custom layers do not define `get_config`.
  Trying to retrieve the config results in errors.

  Args:
    model: A Functional model.
    input_layers: Dictionary mapping input layers in `model` to new input layers
    layer_fn: Function used to clone all non-input layers.

  Returns:
    Model config object, and a dictionary of newly created layers.
  """
    created_layers = {}

    def _copy_layer(layer):
        if layer in input_layers:
            created_layers[layer.name] = input_layers[layer]
        elif layer in model._input_layers:
            created_layers[layer.name] = InputLayer(**layer.get_config())
        else:
            created_layers[layer.name] = layer_fn(layer)
        return {}
    config = functional.get_network_config(model, serialize_layer_fn=_copy_layer)
    return (config, created_layers)