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
def _make_new_nodes(nodes_by_depth, layer_fn, layer_map, tensor_map):
    """Uses the layers in `layer_map` to make new nodes based on `nodes_by_depth`.

  Args:
    nodes_by_depth: Provides structure information to create new nodes.
    layer_fn: Function to clone layers.
    layer_map: Map from layers in `model` to new layers.
    tensor_map: Map from tensors in `model` to newly compute tensors.

  Returns:
    A set of new nodes. `layer_map` and `tensor_map` are updated.
  """
    new_nodes = set()
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = nodes_by_depth[depth]
        for node in nodes:
            layer = node.outbound_layer
            if layer not in layer_map:
                new_layer = layer_fn(layer)
                layer_map[layer] = new_layer
                layer = new_layer
            else:
                layer = layer_map[layer]
                if isinstance(layer, InputLayer):
                    continue
            if all((tensor in tensor_map for tensor in nest.flatten(node.input_tensors))):
                args = nest.map_structure(lambda t: tensor_map.get(t, t), node.call_args)
                kwargs = nest.map_structure(lambda t: tensor_map.get(t, t), node.call_kwargs)
                output_tensors = layer(*args, **kwargs)
                first_output_tensor = nest.flatten(output_tensors)[0]
                new_nodes.add(layer._inbound_nodes[first_output_tensor._keras_history.node_index])
                for x, y in zip(nest.flatten(node.output_tensors), nest.flatten(output_tensors)):
                    tensor_map[x] = y
    return new_nodes