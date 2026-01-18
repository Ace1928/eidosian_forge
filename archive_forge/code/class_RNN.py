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
class RNN(Layer):
    """Base class for recurrent layers.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Args:
    cell: A RNN cell instance or a list of RNN cell instances.
      A RNN cell is a class that has:
      - A `call(input_at_t, states_at_t)` method, returning
        `(output_at_t, states_at_t_plus_1)`. The call method of the
        cell can also take the optional argument `constants`, see
        section "Note on passing external constants" below.
      - A `state_size` attribute. This can be a single integer
        (single state) in which case it is the size of the recurrent
        state. This can also be a list/tuple of integers (one size per state).
        The `state_size` can also be TensorShape or tuple/list of
        TensorShape, to represent high dimension state.
      - A `output_size` attribute. This can be a single integer or a
        TensorShape, which represent the shape of the output. For backward
        compatible reason, if this attribute is not available for the
        cell, the value will be inferred by the first element of the
        `state_size`.
      - A `get_initial_state(inputs=None, batch_size=None, dtype=None)`
        method that creates a tensor meant to be fed to `call()` as the
        initial state, if the user didn't specify any initial state via other
        means. The returned initial state should have a shape of
        [batch_size, cell.state_size]. The cell might choose to create a
        tensor full of zeros, or full of other values based on the cell's
        implementation.
        `inputs` is the input tensor to the RNN layer, which should
        contain the batch size as its shape[0], and also dtype. Note that
        the shape[0] might be `None` during the graph construction. Either
        the `inputs` or the pair of `batch_size` and `dtype` are provided.
        `batch_size` is a scalar tensor that represents the batch size
        of the inputs. `dtype` is `tf.DType` that represents the dtype of
        the inputs.
        For backward compatibility, if this method is not implemented
        by the cell, the RNN layer will create a zero filled tensor with the
        size of [batch_size, cell.state_size].
      In the case that `cell` is a list of RNN cell instances, the cells
      will be stacked on top of each other in the RNN, resulting in an
      efficient stacked RNN.
    return_sequences: Boolean (default `False`). Whether to return the last
      output in the output sequence, or the full sequence.
    return_state: Boolean (default `False`). Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default `False`).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default `False`). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default `False`).
      If True, the network will be unrolled, else a symbolic loop will be used.
      Unrolling can speed-up a RNN, although it tends to be more
      memory-intensive. Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    zero_output_for_mask: Boolean (default `False`).
      Whether the output should use zeros for the masked timesteps. Note that
      this field is only used when `return_sequences` is True and mask is
      provided. It can useful if you want to reuse the raw output sequence of
      the RNN without interference from the masked timesteps, eg, merging
      bidirectional RNNs.

  Call arguments:
    inputs: Input tensor.
    mask: Binary tensor of shape `[batch_size, timesteps]` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False`
      entry indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is for use with cells that use dropout.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
    constants: List of constant tensors to be passed to the cell at each
      timestep.

  Input shape:
    N-D tensor with shape `[batch_size, timesteps, ...]` or
    `[timesteps, batch_size, ...]` when time_major is True.

  Output shape:
    - If `return_state`: a list of tensors. The first tensor is
      the output. The remaining tensors are the last states,
      each with shape `[batch_size, state_size]`, where `state_size` could
      be a high dimension tensor shape.
    - If `return_sequences`: N-D tensor with shape
      `[batch_size, timesteps, output_size]`, where `output_size` could
      be a high dimension tensor shape, or
      `[timesteps, batch_size, output_size]` when `time_major` is True.
    - Else, N-D tensor with shape `[batch_size, output_size]`, where
      `output_size` could be a high dimension tensor shape.

  Masking:
    This layer supports masking for input data with a variable number
    of timesteps. To introduce masks to your data,
    use an [tf.keras.layers.Embedding] layer with the `mask_zero` parameter
    set to `True`.

  Note on using statefulness in RNNs:
    You can set RNN layers to be 'stateful', which means that the states
    computed for the samples in one batch will be reused as initial states
    for the samples in the next batch. This assumes a one-to-one mapping
    between samples in different successive batches.

    To enable statefulness:
      - Specify `stateful=True` in the layer constructor.
      - Specify a fixed batch size for your model, by passing
        If sequential model:
          `batch_input_shape=(...)` to the first layer in your model.
        Else for functional model with 1 or more Input layers:
          `batch_shape=(...)` to all the first layers in your model.
        This is the expected shape of your inputs
        *including the batch size*.
        It should be a tuple of integers, e.g. `(32, 10, 100)`.
      - Specify `shuffle=False` when calling `fit()`.

    To reset the states of your model, call `.reset_states()` on either
    a specific layer, or on your entire model.

  Note on specifying the initial state of RNNs:
    You can specify the initial state of RNN layers symbolically by
    calling them with the keyword argument `initial_state`. The value of
    `initial_state` should be a tensor or list of tensors representing
    the initial state of the RNN layer.

    You can specify the initial state of RNN layers numerically by
    calling `reset_states` with the keyword argument `states`. The value of
    `states` should be a numpy array or list of numpy arrays representing
    the initial state of the RNN layer.

  Note on passing external constants to RNNs:
    You can pass "external" constants to the cell using the `constants`
    keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
    requires that the `cell.call` method accepts the same keyword argument
    `constants`. Such constants can be used to condition the cell
    transformation on additional static inputs (not changing over time),
    a.k.a. an attention mechanism.

  Examples:

  ```python
  # First, let's define a RNN Cell, as a layer subclass.

  class MinimalRNNCell(keras.layers.Layer):

      def __init__(self, units, **kwargs):
          self.units = units
          self.state_size = units
          super(MinimalRNNCell, self).__init__(**kwargs)

      def build(self, input_shape):
          self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer='uniform',
                                        name='kernel')
          self.recurrent_kernel = self.add_weight(
              shape=(self.units, self.units),
              initializer='uniform',
              name='recurrent_kernel')
          self.built = True

      def call(self, inputs, states):
          prev_output = states[0]
          h = backend.dot(inputs, self.kernel)
          output = h + backend.dot(prev_output, self.recurrent_kernel)
          return output, [output]

  # Let's use this cell in a RNN layer:

  cell = MinimalRNNCell(32)
  x = keras.Input((None, 5))
  layer = RNN(cell)
  y = layer(x)

  # Here's how to use the cell to build a stacked RNN:

  cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
  x = keras.Input((None, 5))
  layer = RNN(cells)
  y = layer(x)
  ```
  """

    def __init__(self, cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, time_major=False, **kwargs):
        if isinstance(cell, (list, tuple)):
            cell = StackedRNNCells(cell)
        if not 'call' in dir(cell):
            raise ValueError('`cell` should have a `call` method. The RNN was passed:', cell)
        if not 'state_size' in dir(cell):
            raise ValueError('The RNN cell should have an attribute `state_size` (tuple of integers, one integer per RNN state).')
        self.zero_output_for_mask = kwargs.pop('zero_output_for_mask', False)
        if 'input_shape' not in kwargs and ('input_dim' in kwargs or 'input_length' in kwargs):
            input_shape = (kwargs.pop('input_length', None), kwargs.pop('input_dim', None))
            kwargs['input_shape'] = input_shape
        super(RNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major
        self.supports_masking = True
        self.input_spec = None
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = 0
        if stateful:
            if distribute_lib.has_strategy():
                raise ValueError('RNNs with stateful=True not yet supported with tf.distribute.Strategy.')

    @property
    def _use_input_spec_as_call_signature(self):
        if self.unroll:
            return False
        return super(RNN, self)._use_input_spec_as_call_signature

    @property
    def states(self):
        if self._states is None:
            state = nest.map_structure(lambda _: None, self.cell.state_size)
            return state if nest.is_nested(self.cell.state_size) else [state]
        return self._states

    @states.setter
    @trackable.no_automatic_dependency_tracking
    def states(self, states):
        self._states = states

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        try:
            input_shape = tensor_shape.TensorShape(input_shape)
        except (ValueError, TypeError):
            input_shape = nest.flatten(input_shape)[0]
        batch = input_shape[0]
        time_step = input_shape[1]
        if self.time_major:
            batch, time_step = (time_step, batch)
        if _is_multiple_state(self.cell.state_size):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]

        def _get_output_shape(flat_output_size):
            output_dim = tensor_shape.TensorShape(flat_output_size).as_list()
            if self.return_sequences:
                if self.time_major:
                    output_shape = tensor_shape.TensorShape([time_step, batch] + output_dim)
                else:
                    output_shape = tensor_shape.TensorShape([batch, time_step] + output_dim)
            else:
                output_shape = tensor_shape.TensorShape([batch] + output_dim)
            return output_shape
        if getattr(self.cell, 'output_size', None) is not None:
            output_shape = nest.flatten(nest.map_structure(_get_output_shape, self.cell.output_size))
            output_shape = output_shape[0] if len(output_shape) == 1 else output_shape
        else:
            output_shape = _get_output_shape(state_size[0])
        if self.return_state:

            def _get_state_shape(flat_state):
                state_shape = [batch] + tensor_shape.TensorShape(flat_state).as_list()
                return tensor_shape.TensorShape(state_shape)
            state_shape = nest.map_structure(_get_state_shape, state_size)
            return generic_utils.to_list(output_shape) + nest.flatten(state_shape)
        else:
            return output_shape

    def compute_mask(self, inputs, mask):
        mask = nest.flatten(mask)[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        def get_input_spec(shape):
            """Convert input shape to InputSpec."""
            if isinstance(shape, tensor_shape.TensorShape):
                input_spec_shape = shape.as_list()
            else:
                input_spec_shape = list(shape)
            batch_index, time_step_index = (1, 0) if self.time_major else (0, 1)
            if not self.stateful:
                input_spec_shape[batch_index] = None
            input_spec_shape[time_step_index] = None
            return InputSpec(shape=tuple(input_spec_shape))

        def get_step_input_shape(shape):
            if isinstance(shape, tensor_shape.TensorShape):
                shape = tuple(shape.as_list())
            return shape[1:] if self.time_major else (shape[0],) + shape[2:]
        try:
            input_shape = tensor_shape.TensorShape(input_shape)
        except (ValueError, TypeError):
            pass
        if not nest.is_nested(input_shape):
            if self.input_spec is not None:
                self.input_spec[0] = get_input_spec(input_shape)
            else:
                self.input_spec = [get_input_spec(input_shape)]
            step_input_shape = get_step_input_shape(input_shape)
        else:
            if self.input_spec is not None:
                self.input_spec[0] = nest.map_structure(get_input_spec, input_shape)
            else:
                self.input_spec = generic_utils.to_list(nest.map_structure(get_input_spec, input_shape))
            step_input_shape = nest.map_structure(get_step_input_shape, input_shape)
        if isinstance(self.cell, Layer) and (not self.cell.built):
            with backend.name_scope(self.cell.name):
                self.cell.build(step_input_shape)
                self.cell.built = True
        if _is_multiple_state(self.cell.state_size):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]
        if self.state_spec is not None:
            self._validate_state_spec(state_size, self.state_spec)
        else:
            self.state_spec = [InputSpec(shape=[None] + tensor_shape.TensorShape(dim).as_list()) for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    @staticmethod
    def _validate_state_spec(cell_state_sizes, init_state_specs):
        """Validate the state spec between the initial_state and the state_size.

    Args:
      cell_state_sizes: list, the `state_size` attribute from the cell.
      init_state_specs: list, the `state_spec` from the initial_state that is
        passed in `call()`.

    Raises:
      ValueError: When initial state spec is not compatible with the state size.
    """
        validation_error = ValueError('An `initial_state` was passed that is not compatible with `cell.state_size`. Received `state_spec`={}; however `cell.state_size` is {}'.format(init_state_specs, cell_state_sizes))
        flat_cell_state_sizes = nest.flatten(cell_state_sizes)
        flat_state_specs = nest.flatten(init_state_specs)
        if len(flat_cell_state_sizes) != len(flat_state_specs):
            raise validation_error
        for cell_state_spec, cell_state_size in zip(flat_state_specs, flat_cell_state_sizes):
            if not tensor_shape.TensorShape(cell_state_spec.shape[1:]).is_compatible_with(tensor_shape.TensorShape(cell_state_size)):
                raise validation_error

    @doc_controls.do_not_doc_inheritable
    def get_initial_state(self, inputs):
        get_initial_state_fn = getattr(self.cell, 'get_initial_state', None)
        if nest.is_nested(inputs):
            inputs = nest.flatten(inputs)[0]
        input_shape = array_ops.shape(inputs)
        batch_size = input_shape[1] if self.time_major else input_shape[0]
        dtype = inputs.dtype
        if get_initial_state_fn:
            init_state = get_initial_state_fn(inputs=None, batch_size=batch_size, dtype=dtype)
        else:
            init_state = _generate_zero_filled_state(batch_size, self.cell.state_size, dtype)
        if not nest.is_nested(init_state):
            init_state = [init_state]
        return list(init_state)

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(inputs, initial_state, constants, self._num_constants)
        if initial_state is None and constants is None:
            return super(RNN, self).__call__(inputs, **kwargs)
        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            additional_inputs += initial_state
            self.state_spec = nest.map_structure(lambda s: InputSpec(shape=backend.int_shape(s)), initial_state)
            additional_specs += self.state_spec
        if constants is not None:
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=backend.int_shape(constant)) for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        flat_additional_inputs = nest.flatten(additional_inputs)
        is_keras_tensor = backend.is_keras_tensor(flat_additional_inputs[0]) if flat_additional_inputs else True
        for tensor in flat_additional_inputs:
            if backend.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN layer cannot be specified with a mix of Keras tensors and non-Keras tensors (a "Keras tensor" is a tensor that was returned by a Keras layer, or by `Input`)')
        if is_keras_tensor:
            full_input = [inputs] + additional_inputs
            if self.built:
                full_input_spec = self.input_spec + additional_specs
            else:
                full_input_spec = generic_utils.to_list(nest.map_structure(lambda _: None, inputs)) + additional_specs
            self.input_spec = full_input_spec
            output = super(RNN, self).__call__(full_input, **kwargs)
            self.input_spec = self.input_spec[:-len(additional_specs)]
            return output
        else:
            if initial_state is not None:
                kwargs['initial_state'] = initial_state
            if constants is not None:
                kwargs['constants'] = constants
            return super(RNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
        is_ragged_input = row_lengths is not None
        self._validate_args_if_ragged(is_ragged_input, mask)
        inputs, initial_state, constants = self._process_inputs(inputs, initial_state, constants)
        self._maybe_reset_cell_dropout_mask(self.cell)
        if isinstance(self.cell, StackedRNNCells):
            for cell in self.cell.cells:
                self._maybe_reset_cell_dropout_mask(cell)
        if mask is not None:
            mask = nest.flatten(mask)[0]
        if nest.is_nested(inputs):
            input_shape = backend.int_shape(nest.flatten(inputs)[0])
        else:
            input_shape = backend.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]
        if self.unroll and timesteps is None:
            raise ValueError('Cannot unroll a RNN if the time dimension is undefined. \n- If using a Sequential model, specify the time dimension by passing an `input_shape` or `batch_input_shape` argument to your first layer. If your first layer is an Embedding, you can also use the `input_length` argument.\n- If using the functional API, specify the time dimension by passing a `shape` or `batch_shape` argument to your Input layer.')
        kwargs = {}
        if generic_utils.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training
        is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
        cell_call_fn = self.cell.__call__ if callable(self.cell) else self.cell.call
        if constants:
            if not generic_utils.has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                output, new_states = cell_call_fn(inputs, states, constants=constants, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return (output, new_states)
        else:

            def step(inputs, states):
                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                output, new_states = cell_call_fn(inputs, states, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return (output, new_states)
        last_output, outputs, states = backend.rnn(step, inputs, initial_state, constants=constants, go_backwards=self.go_backwards, mask=mask, unroll=self.unroll, input_length=row_lengths if row_lengths is not None else timesteps, time_major=self.time_major, zero_output_for_mask=self.zero_output_for_mask)
        if self.stateful:
            updates = [state_ops.assign(self_state, state) for self_state, state in zip(nest.flatten(self.states), nest.flatten(states))]
            self.add_update(updates)
        if self.return_sequences:
            output = backend.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards)
        else:
            output = last_output
        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return generic_utils.to_list(output) + states
        else:
            return output

    def _process_inputs(self, inputs, initial_state, constants):
        if isinstance(inputs, collections.abc.Sequence) and (not isinstance(inputs, tuple)):
            if not self._num_constants:
                initial_state = inputs[1:]
            else:
                initial_state = inputs[1:-self._num_constants]
                constants = inputs[-self._num_constants:]
            if len(initial_state) == 0:
                initial_state = None
            inputs = inputs[0]
        if self.stateful:
            if initial_state is not None:
                non_zero_count = math_ops.add_n([math_ops.count_nonzero_v2(s) for s in nest.flatten(self.states)])
                initial_state = cond.cond(non_zero_count > 0, true_fn=lambda: self.states, false_fn=lambda: initial_state, strict=True)
            else:
                initial_state = self.states
        elif initial_state is None:
            initial_state = self.get_initial_state(inputs)
        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) + ' states but was passed ' + str(len(initial_state)) + ' initial states.')
        return (inputs, initial_state, constants)

    def _validate_args_if_ragged(self, is_ragged_input, mask):
        if not is_ragged_input:
            return
        if mask is not None:
            raise ValueError('The mask that was passed in was ' + str(mask) + ' and cannot be applied to RaggedTensor inputs. Please make sure that there is no mask passed in by upstream layers.')
        if self.unroll:
            raise ValueError('The input received contains RaggedTensors and does not support unrolling. Disable unrolling by passing `unroll=False` in the RNN Layer constructor.')

    def _maybe_reset_cell_dropout_mask(self, cell):
        if isinstance(cell, DropoutRNNCellMixin):
            cell.reset_dropout_mask()
            cell.reset_recurrent_dropout_mask()

    def reset_states(self, states=None):
        """Reset the recorded states for the stateful RNN layer.

    Can only be used when RNN layer is constructed with `stateful` = `True`.
    Args:
      states: Numpy arrays that contains the value for the initial state, which
        will be feed to cell at the first time step. When the value is None,
        zero filled numpy array will be created based on the cell state size.

    Raises:
      AttributeError: When the RNN layer is not stateful.
      ValueError: When the batch size of the RNN layer is unknown.
      ValueError: When the input numpy array is not compatible with the RNN
        layer state, either size wise or dtype wise.
    """
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        spec_shape = None
        if self.input_spec is not None:
            spec_shape = nest.flatten(self.input_spec[0])[0].shape
        if spec_shape is None:
            batch_size = None
        else:
            batch_size = spec_shape[1] if self.time_major else spec_shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batch_input_shape` argument to your first layer.\n- If using the functional API, specify the batch size by passing a `batch_shape` argument to your Input layer.')
        if nest.flatten(self.states)[0] is None:
            if getattr(self.cell, 'get_initial_state', None):
                flat_init_state_values = nest.flatten(self.cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=self.dtype or backend.floatx()))
            else:
                flat_init_state_values = nest.flatten(_generate_zero_filled_state(batch_size, self.cell.state_size, self.dtype or backend.floatx()))
            flat_states_variables = nest.map_structure(backend.variable, flat_init_state_values)
            self.states = nest.pack_sequence_as(self.cell.state_size, flat_states_variables)
            if not nest.is_nested(self.states):
                self.states = [self.states]
        elif states is None:
            for state, size in zip(nest.flatten(self.states), nest.flatten(self.cell.state_size)):
                backend.set_value(state, np.zeros([batch_size] + tensor_shape.TensorShape(size).as_list()))
        else:
            flat_states = nest.flatten(self.states)
            flat_input_states = nest.flatten(states)
            if len(flat_input_states) != len(flat_states):
                raise ValueError('Layer ' + self.name + ' expects ' + str(len(flat_states)) + ' states, but it received ' + str(len(flat_input_states)) + ' state values. Input received: ' + str(states))
            set_value_tuples = []
            for i, (value, state) in enumerate(zip(flat_input_states, flat_states)):
                if value.shape != state.shape:
                    raise ValueError('State ' + str(i) + ' is incompatible with layer ' + self.name + ': expected shape=' + str((batch_size, state)) + ', found shape=' + str(value.shape))
                set_value_tuples.append((state, value))
            backend.batch_set_value(set_value_tuples)

    def get_config(self):
        config = {'return_sequences': self.return_sequences, 'return_state': self.return_state, 'go_backwards': self.go_backwards, 'stateful': self.stateful, 'unroll': self.unroll, 'time_major': self.time_major}
        if self._num_constants:
            config['num_constants'] = self._num_constants
        if self.zero_output_for_mask:
            config['zero_output_for_mask'] = self.zero_output_for_mask
        config['cell'] = generic_utils.serialize_keras_object(self.cell)
        base_config = super(RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.keras.layers import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'), custom_objects=custom_objects)
        num_constants = config.pop('num_constants', 0)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer

    @property
    def _trackable_saved_model_saver(self):
        return layer_serialization.RNNSavedModelSaver(self)