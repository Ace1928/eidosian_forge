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
def _clone_functional_model(model, input_tensors=None, layer_fn=_clone_layer):
    """Clone a functional `Model` instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Input layers are always cloned.

  Args:
      model: Instance of `Model`.
      input_tensors: optional list of input tensors
          to build the model upon. If not provided,
          placeholders will be created.
      layer_fn: callable to be applied on non-input layers in the model. By
          default it clones the layer. Another example is to preserve the layer
          to share the weights. This is required when we create a per-replica
          copy of the model with distribution strategy; we want the weights to
          be shared but still feed inputs separately so we create new input
          layers.

  Returns:
      An instance of `Model` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights.

  Raises:
      ValueError: in case of invalid `model` argument value or `layer_fn`
      argument value.
  """
    if not isinstance(model, Model):
        raise ValueError('Expected `model` argument to be a `Model` instance, got ', model)
    if isinstance(model, Sequential):
        raise ValueError('Expected `model` argument to be a functional `Model` instance, got a `Sequential` instance instead:', model)
    if not model._is_graph_network:
        raise ValueError('Expected `model` argument to be a functional `Model` instance, but got a subclass model instead.')
    new_input_layers = {}
    if input_tensors is not None:
        input_tensors = nest.flatten(input_tensors)
        for i, input_tensor in enumerate(input_tensors):
            original_input_layer = model._input_layers[i]
            if not backend.is_keras_tensor(input_tensor):
                name = original_input_layer.name
                input_tensor = Input(tensor=input_tensor, name='input_wrapper_for_' + name)
                newly_created_input_layer = input_tensor._keras_history.layer
                new_input_layers[original_input_layer] = newly_created_input_layer
            else:
                new_input_layers[original_input_layer] = original_input_layer
    if not callable(layer_fn):
        raise ValueError('Expected `layer_fn` argument to be a callable.')
    model_configs, created_layers = _clone_layers_and_model_config(model, new_input_layers, layer_fn)
    input_tensors, output_tensors, created_layers = functional.reconstruct_from_config(model_configs, created_layers=created_layers)
    metrics_names = model.metrics_names
    model = Model(input_tensors, output_tensors, name=model.name)
    ancillary_layers = [layer for layer in created_layers.values() if layer not in model.layers]
    if ancillary_layers:
        new_nodes = nest.flatten([layer.inbound_nodes[1:] if functional._should_skip_first_node(layer) else layer.inbound_nodes for layer in created_layers.values()])
        _insert_ancillary_layers(model, ancillary_layers, metrics_names, new_nodes)
    return model