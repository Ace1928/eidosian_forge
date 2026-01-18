import collections
import warnings
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class StackedRNNCells(Layer):
    """Wrapper allowing a stack of RNN cells to behave as a single cell.

  Used to implement efficient stacked RNNs.

  Args:
    cells: List of RNN cell instances.

  Examples:

  ```python
  batch_size = 3
  sentence_max_length = 5
  n_features = 2
  new_shape = (batch_size, sentence_max_length, n_features)
  x = tf.constant(np.reshape(np.arange(30), new_shape), dtype = tf.float32)

  rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(2)]
  stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
  lstm_layer = tf.keras.layers.RNN(stacked_lstm)

  result = lstm_layer(x)
  ```
  """

    def __init__(self, cells, **kwargs):
        for cell in cells:
            if not 'call' in dir(cell):
                raise ValueError('All cells must have a `call` method. received cells:', cells)
            if not 'state_size' in dir(cell):
                raise ValueError('All cells must have a `state_size` attribute. received cells:', cells)
        self.cells = cells
        self.reverse_state_order = kwargs.pop('reverse_state_order', False)
        if self.reverse_state_order:
            logging.warning('reverse_state_order=True in StackedRNNCells will soon be deprecated. Please update the code to work with the natural order of states if you rely on the RNN states, eg RNN(return_state=True).')
        super(StackedRNNCells, self).__init__(**kwargs)

    @property
    def state_size(self):
        return tuple((c.state_size for c in (self.cells[::-1] if self.reverse_state_order else self.cells)))

    @property
    def output_size(self):
        if getattr(self.cells[-1], 'output_size', None) is not None:
            return self.cells[-1].output_size
        elif _is_multiple_state(self.cells[-1].state_size):
            return self.cells[-1].state_size[0]
        else:
            return self.cells[-1].state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_states = []
        for cell in self.cells[::-1] if self.reverse_state_order else self.cells:
            get_initial_state_fn = getattr(cell, 'get_initial_state', None)
            if get_initial_state_fn:
                initial_states.append(get_initial_state_fn(inputs=inputs, batch_size=batch_size, dtype=dtype))
            else:
                initial_states.append(_generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype))
        return tuple(initial_states)

    def call(self, inputs, states, constants=None, training=None, **kwargs):
        state_size = self.state_size[::-1] if self.reverse_state_order else self.state_size
        nested_states = nest.pack_sequence_as(state_size, nest.flatten(states))
        new_nested_states = []
        for cell, states in zip(self.cells, nested_states):
            states = states if nest.is_nested(states) else [states]
            is_tf_rnn_cell = getattr(cell, '_is_tf_rnn_cell', None) is not None
            states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
            if generic_utils.has_arg(cell.call, 'training'):
                kwargs['training'] = training
            else:
                kwargs.pop('training', None)
            cell_call_fn = cell.__call__ if callable(cell) else cell.call
            if generic_utils.has_arg(cell.call, 'constants'):
                inputs, states = cell_call_fn(inputs, states, constants=constants, **kwargs)
            else:
                inputs, states = cell_call_fn(inputs, states, **kwargs)
            new_nested_states.append(states)
        return (inputs, nest.pack_sequence_as(state_size, nest.flatten(new_nested_states)))

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        for cell in self.cells:
            if isinstance(cell, Layer) and (not cell.built):
                with backend.name_scope(cell.name):
                    cell.build(input_shape)
                    cell.built = True
            if getattr(cell, 'output_size', None) is not None:
                output_dim = cell.output_size
            elif _is_multiple_state(cell.state_size):
                output_dim = cell.state_size[0]
            else:
                output_dim = cell.state_size
            input_shape = tuple([input_shape[0]] + tensor_shape.TensorShape(output_dim).as_list())
        self.built = True

    def get_config(self):
        cells = []
        for cell in self.cells:
            cells.append(generic_utils.serialize_keras_object(cell))
        config = {'cells': cells}
        base_config = super(StackedRNNCells, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.keras.layers import deserialize as deserialize_layer
        cells = []
        for cell_config in config.pop('cells'):
            cells.append(deserialize_layer(cell_config, custom_objects=custom_objects))
        return cls(cells, **config)