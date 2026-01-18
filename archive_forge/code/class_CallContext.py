import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
class CallContext(object):
    """Keeps track of properties currently inside a Layer/Model's `call`.

  Attributes:
    in_call: Whether currently inside the `call` of a Layer.
    layer: The `Layer` whose `call` is currently active.
    inputs: The inputs to the currently active `Layer`.
    build_graph: Whether currently inside a Graph or FuncGraph.
    training: Whether currently executing in training or inference mode.
    saving: Whether currently saving to SavedModel.
    frozen: Whether currently executing inside a `Layer` with `trainable` set to
      `False`.
    in_keras_graph: Whether executing inside the Keras Graph.
  """

    def __init__(self):
        self.in_call = False
        self._state = {'layer': None, 'inputs': None, 'build_graph': False, 'training': None, 'saving': None}
        self._in_keras_graph = False

    def enter(self, layer, inputs, build_graph, training, saving=None):
        """Push a Layer and its inputs and state onto the current call context.

    Args:
      layer: The `Layer` whose `call` is currently active.
      inputs: The inputs to the currently active `Layer`.
      build_graph: Whether currently inside a Graph or FuncGraph.
      training: Whether currently executing in training or inference mode.
      saving: Whether currently saving to SavedModel.

    Returns:
      Context manager.
    """
        state = {'layer': layer, 'inputs': inputs, 'build_graph': build_graph, 'training': training, 'saving': saving}
        return CallContextManager(self, state)

    @property
    def layer(self):
        return self._state['layer']

    @property
    def inputs(self):
        return self._state['inputs']

    @property
    def build_graph(self):
        return self._state['build_graph']

    @property
    def training(self):
        return self._state['training']

    @property
    def saving(self):
        return self._state['saving']

    @property
    def frozen(self):
        layer = self._state['layer']
        if not layer:
            return False
        return not layer.trainable

    @property
    def in_keras_graph(self):
        if context.executing_eagerly():
            return False
        return self._in_keras_graph or getattr(backend.get_graph(), 'name', None) == 'keras_graph'