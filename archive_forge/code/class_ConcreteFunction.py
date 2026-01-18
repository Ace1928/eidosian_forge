import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
class ConcreteFunction(core.ConcreteFunction, trackable.Trackable):
    """A `tf.types.experimental.ConcreteFunction` created from `tf.function`."""

    def __init__(self, atomic_fn, shared_func_graph=True):
        """Initialize a `ConcreteFunction`.

    Args:
     atomic_fn: Inference atomic function to form basis of forward pass.
     shared_func_graph: If False, the ConcreteFunction takes ownership of
       `func_graph` and will break reference cycles when it is deleted. This
       makes the FuncGraph inoperable.

    Raises:
      ValueError: If number of input_placeholders is not equal to the number
        of function inputs.
    """
        self._arg_keywords = None
        self._num_positional_args = None
        self._func_graph = atomic_fn.graph
        self._captured_inputs = self._func_graph.external_captures + self._func_graph.deferred_external_captures
        self._function_type = atomic_fn.function_type
        self._output_shapes = tuple((output.shape for output in self._func_graph.outputs))
        self._attrs = attributes_lib.parse_func_attrs(atomic_fn.attributes or {})
        if shared_func_graph:
            self._garbage_collector = None
        else:
            self._garbage_collector = ConcreteFunctionGarbageCollector(atomic_fn.graph)
        self._delayed_rewrite_functions = _DelayedRewriteGradientFunctions(atomic_fn, self._garbage_collector)
        self._first_order_tape_functions = {}
        self._higher_order_tape_functions = {}
        self._inference_function = self._delayed_rewrite_functions.forward()

    @classmethod
    def from_func_graph(cls, graph, function_type, attrs, shared_func_graph=True):
        atomic_fn = atomic_function.from_func_graph(_inference_name(graph.name), graph, attrs, function_type)
        return ConcreteFunction(atomic_fn, shared_func_graph=shared_func_graph)

    @property
    def function_type(self):
        """Return the FunctionType associated with this ConcreteFunction."""
        return self._function_type

    @property
    def _function_spec(self):
        if self.function_type is None:
            return None
        return function_type_utils.FunctionSpec(self.function_type, {p.default for p in self.function_type.parameters.values() if p.optional}, False, name=self.name)

    @property
    def variables(self):
        """Sequence of variables for this function."""
        return tuple(self._func_graph.variables)

    def set_variables(self, variables):
        self._func_graph.variables = variables

    @property
    def trainable_variables(self):
        """Sequence of trainable variables for this function."""
        return tuple(self._func_graph.trainable_variables)

    def __call__(self, *args, **kwargs):
        """Executes the wrapped function.

    ConcreteFunctions have two signatures:

    * The signature of the original function wrapped by this ConcreteFunction.
    * A flat signature, where each argument accepts a single Tensor.

    The original function signature is generally preferred, but the flat input
    signature is supported for backward compatibility.

    ### Original Function Signature

    When calling a ConcreteFunction with the signature of the original function,
    each argument must match the type or value that was used when the
    ConcreteFunction's graph was traced.  In particular:

    * Tensor arguments (including CompositeTensors, such as RaggedTensor) must
      have matching `TypeSpec`s.
    * Non-Tensor arguments (such as booleans or ints) must have equal values.
    * Nested arguments (such as lists, tuples, or dictionaries) must have the
      same nesting structure; and each nested value must have a matching type
      or value.

    The default value for any arguments that were traced with non-Tensor values
    is the value that was used in the trace.  Arguments that were traced with
    tensor arguments do not have a default value (even if the original function
    had a default value for that argument).

    ### Flat Signature

    When calling a ConcreteFunction with the flat signature, the arguments
    correspond to the flattened component tensors of the arguments that were
    used to construct the ConcreteFunction.  Parameter names are assigned based
    on `TensorSpec.name` (when specified) or the original argument names (with
    suffixes automatically added for nested arguments or composite tensors with
    multiple components).

    Args:
      *args: Positional arguments to the concrete function.
      **kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the TF function on the given Tensors.

    Raises:
      AssertionError: If this `ConcreteFunction` was not created through
        `get_concrete_function`.
      TypeError: If the arguments do not match the function's signature.
    """
        return self._call_impl(args, kwargs)

    def _call_impl(self, args, kwargs):
        """See `__call__` for details."""
        with trace.Trace(self._func_graph.name, tf_function_call='concrete'):
            if self.function_type is not None:
                try:
                    return self._call_with_structured_signature(args, kwargs)
                except TypeError as structured_err:
                    try:
                        return self._call_with_flat_signature(args, kwargs)
                    except (TypeError, ValueError) as flat_err:
                        raise TypeError(str(structured_err) + '\nFallback to flat signature also failed due to: ' + str(flat_err))
            return self._call_with_flat_signature(args, kwargs)

    def _call_with_flat_signature(self, args, kwargs):
        """Executes the wrapped function with the flat signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the flat signature of this
        `ConcreteFunction`.
    """
        if len(args) > self._num_positional_args:
            raise TypeError(f'{self._flat_signature_summary()} takes {self._num_positional_args} positional arguments, got {len(args)}.')
        args = list(args)
        kwargs = dict(kwargs)
        kwargs = {function_type_lib.sanitize_arg_name(k): v for k, v in kwargs.items()}
        for keyword in self._arg_keywords[len(args):]:
            try:
                args.append(kwargs.pop(function_type_lib.sanitize_arg_name(compat.as_str(keyword))))
            except KeyError:
                specified_keywords = list(self._arg_keywords[:len(args)]) + list(kwargs.keys())
                missing_required_args = sorted(set(self._arg_keywords) - set(specified_keywords))
                raise TypeError(f'{self._flat_signature_summary()} missing required arguments: {', '.join(missing_required_args)}.')
        if kwargs:
            positional_arg_keywords = set(self._arg_keywords[:len(args)])
            for unused_key in kwargs:
                if unused_key in positional_arg_keywords:
                    raise TypeError(f"{self._flat_signature_summary()} got two values for '{unused_key}'.")
            raise TypeError(f'{self._flat_signature_summary()} got unexpected keyword arguments: {', '.join(sorted(kwargs))}.')
        for i, arg in enumerate(args):
            if not isinstance(arg, (tensor_lib.Tensor, resource_variable_ops.BaseResourceVariable)):
                raise TypeError(f'{self._flat_signature_summary()}: expected argument #{i}(zero-based) to be a Tensor; got {type(arg).__name__} ({arg}).')
        return self._call_flat(args, self.captured_inputs)

    def _call_with_structured_signature(self, args, kwargs):
        """Executes the wrapped function with the structured signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the structured signature
        of this `ConcreteFunction`.
    """
        bound_args = function_type_utils.canonicalize_function_inputs(args, kwargs, self.function_type)
        filtered_flat_args = self.function_type.unpack_inputs(bound_args)
        return self._call_flat(filtered_flat_args, captured_inputs=self.captured_inputs)

    def _call_flat(self, tensor_inputs, captured_inputs):
        """Executes the wrapped function.

    Args:
      tensor_inputs: a list of only Tensors generated from args, kwargs.
      captured_inputs: the captured inputs that are also part of the input args
        to the actual execution. By default, it should be self._captured_inputs.
    Returns:
      The result of applying the TF function to `args`.

    Raises:
      ValueError: If `args` contains anything other than Tensors or Variables.
    """
        ctx = context.context()
        executing_eagerly = ctx.executing_eagerly()
        default_graph = ops.get_default_graph()
        if default_graph.building_function and (not self._func_graph.saveable):
            default_graph.mark_as_unsaveable(self._func_graph.saving_errors)
        if record.could_possibly_record() or hasattr(default_graph, 'watch_variable'):
            for v in self._func_graph.variables:
                resource_variable_ops.variable_accessed(v)
        if not executing_eagerly:
            for i, tensor_input in enumerate(tensor_inputs):
                if tensor_input.dtype == dtypes.resource or tensor_input.dtype == dtypes.variant:
                    continue
                graph_input_shape = tensor_shape.TensorShape(self._func_graph.inputs[i].shape)
                if not graph_input_shape.is_compatible_with(tensor_input.shape):
                    raise ValueError(f'Tensor {tensor_input} is not compatible with the shape this function was traced with. Expected shape {self._func_graph.inputs[i].shape}, but got shape {tensor_input.shape}.\n\nIf you called get_concrete_function, you may need to pass a tf.TensorSpec(..., shape=...) with a less specific shape, having None on axes which can vary.')
        args = tensor_inputs + captured_inputs
        possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
        if possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE and executing_eagerly:
            return self._inference_function.flat_call(args)
        forward_backward = self._select_forward_and_backward_functions(args, possible_gradient_type, executing_eagerly)
        forward_function, args_with_tangents = forward_backward.forward()
        if executing_eagerly:
            flat_outputs = forward_function(*args_with_tangents)
        else:
            with default_graph._override_gradient_function({'PartitionedCall': self._get_gradient_function(), 'StatefulPartitionedCall': self._get_gradient_function()}):
                flat_outputs = forward_function(*args_with_tangents)
        forward_backward.record(flat_outputs)
        return self.function_type.pack_output(flat_outputs)

    @property
    def name(self):
        """`ConcreteFunction` name."""
        return self._delayed_rewrite_functions.forward().name

    @property
    def graph(self):
        """Returns the graph from which this function was constructed."""
        return self._func_graph

    @property
    def inputs(self):
        """Returns tensors in `self.graph` corresponding to arguments."""
        return self._func_graph.inputs

    @property
    def structured_input_signature(self):
        """Returns structured signature for this concrete function.

    Returns:
      A tuple `(args, kwargs)`, where:

        * `args` is a tuple that specifies the expected type or value each for
          positional argument.
        * `kwargs` is a dictionary that specifies the expected type or value
          for each keyword-only argument.

      The type or value for each argument is specified using one of the
      following:

        * A `tf.TypeSpec`, indicating that a Tensor or other TensorFlow-native
          value is expected.
        * A Python value, such as an integer, indicating that an equal value
          is expected.
        * A nested structure of `tf.TypeSpec`s and Python values, indicating
          that a corresponding nested structure is expected.
    """
        return self._func_graph.structured_input_signature

    @property
    def outputs(self):
        """Returns tensors in `self.graph` corresponding to returned tensors."""
        return self._func_graph.outputs

    @property
    def structured_outputs(self):
        """Returns outputs in `self.graph` as returned by the original function."""
        return self._func_graph.structured_outputs

    def set_external_captures(self, captures):
        """Updates the function capture values.

    The new values must have tensor types and shapes consistent with the
    original captures of the concrete function, but it is allowed to change a
    value captured with a deferred one and vice-versa.

    Args:
      captures: A list of tensors or closures. Tensors are value captures, and
        closures are call-time (deferred captures).
    """
        self._captured_inputs = captures

    def replace_capture_with_deferred_capture(self, tensor, closure, spec, placeholder=None, default_value=None):
        """Replaces existing capture `tensor` with a deferred capture `closure`.

    This API replaces the capture `tensor` from the concrete function's captured
    inputs list, and places the deferred capture `closure` in
    its spot so the order of captured inputs is preserved. This is important
    because the old `tensor` and the new `closure` will have the same internal
    placeholder, which can be passed through the `placeholder` argument, or
    skipped, in which case we find the placeholder from internal inputs by
    indexing `tensor` in the external captured inputs list. Thus, it is
    important that the new deferred capture has output spec (specified by the
    `spec` argument) compatible with the internal placeholder (`placeholder`)
    and the original capture (`tensor`).

    For example,

    ```python
    bool_captured_tensor = tf.constant(True)
    float_captured_tensor = tf.constant([3.], dtype=tf.float32)
    value = tf.constant([2.], dtype=tf.float32)

    @tf.function
    def fn():
      deferred_tensor = ops.get_default_graph().capture_call_time_value(
          lambda: value,
          tf.TensorSpec(shape=(1,), dtype=tf.float32))
      if bool_captured_tensor:
        return deferred_tensor
      else:
        return deferred_tensor + float_captured_tensor

    concrete_fn = fn.get_concrete_function()
    print(concrete_fn())  # tf.Tensor([2.], shape=(1,), dtype=float32)

    new_bool_captured_tensor = constant_op.constant(False)
    def bool_closure():
      return new_bool_captured_tensor

    concrete_fn.replace_capture_with_deferred_capture(
        bool_captured_tensor,
        bool_closure,
        spec=tensor_lib.TensorSpec(shape=(), dtype=dtypes.bool))

    print(concrete_fn())  # tf.Tensor([5.], shape=(1,), dtype=float32)
    ```

    Args:
      tensor: Tensor already captured. This `tensor` should be listed in
        concrete_function.captured_inputs except when it's empty such as when
        the concrete function is restored from SavedModel.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      placeholder: optional. The internal placeholder corresponding to the
        captured `tensor` and the new `closure`.
      default_value: optional value to use in environments that cannot safely
        evaluate closure.
    """
        capture_index = None
        for i, capture in enumerate(self._captured_inputs):
            if id(tensor) == id(capture):
                capture_index = i
                break
        if placeholder is None:
            if capture_index is None:
                raise ValueError(f"Did not find `tensor` argument {tensor} in the ConcreteFunction's captured inputs list, and did not receive a placeholder argument. Thus we're unable to infer the internal placeholder. ")
            placeholder = self.inputs[-len(self._captured_inputs) + capture_index]
        if not (spec.is_compatible_with(tensor) or spec.is_compatible_with(placeholder)):
            raise ValueError(f"Attempting to substitute closure with spec {spec} that's incompatible with the original capture {tensor} or the internal placeholder {placeholder}.")
        self._func_graph.replace_capture_with_deferred_capture(tensor=tensor, closure=closure, spec=spec, placeholder=placeholder, default_value=default_value)
        if capture_index is not None:
            self._captured_inputs[capture_index] = closure

    @property
    def captured_inputs(self):
        """Returns external Tensors captured by this function.

    self.__call__(*args) passes `args + self.captured_inputs` to the function.
    """
        return nest.flatten([x() if callable(x) else x for x in self._captured_inputs], expand_composites=True)

    @property
    def function_def(self):
        """Returns a `FunctionDef` object representing this function."""
        return self._delayed_rewrite_functions.forward().cached_definition

    @property
    def output_shapes(self):
        """The function's output shapes."""
        return nest.map_structure(lambda x: getattr(x, 'shape', tensor_shape.TensorShape(None)), composite_tensor.replace_composites_with_components(self._func_graph.structured_outputs), expand_composites=False)

    @property
    def output_dtypes(self):
        return nest.map_structure(lambda x: x.dtype if x is not None else None, composite_tensor.replace_composites_with_components(self._func_graph.structured_outputs), expand_composites=False)

    def add_to_graph(self, g=None, overwrite=False):
        """Registers the function, adds it to the graph g or default graph.

    Args:
      g: If specified, registers the function with this graph. Defaults to the
        current context (either the default graph or the eager context).
      overwrite: A bool. If True, its forward function will overwrite
        any existing function of the same signature name in the graph `g`.
    """
        if not context.executing_eagerly() and (not g):
            g = ops.get_default_graph()
        if g is not None:
            g._add_function_recursive(self._delayed_rewrite_functions.forward())

    def add_gradient_functions_to_graph(self, g=None):
        """Add forward/backward functions to graph `g` or the current context."""
        if not context.executing_eagerly() and (not g):
            g = ops.get_default_graph()
        g._add_function_recursive(self._delayed_rewrite_functions.forward())
        forward_function, backward_function = self._delayed_rewrite_functions.forward_backward()
        g._add_function_recursive(forward_function)
        backward_function.add_to_graph(g)

    def _get_gradient_function(self):
        """Returns gradient function. It will be lazily created at first call."""
        return self._delayed_rewrite_functions._rewrite_forward_and_call_backward

    def _select_forward_and_backward_functions(self, args, possible_gradient_type, executing_eagerly):
        """Selects forward and backward functions based on the calling context.

    The forward function computes the "real" function outputs, `self._outputs`,
    and any extra values needed by the corresponding backward function.

    Args:
      args: A flat list of Tensors with all of the inputs to the forward
        function (including user-specified and captured inputs).
      possible_gradient_type: One of gradients_util.POSSIBLE_GRADIENT_TYPES_*.
      executing_eagerly: Boolean, the value of context.executing_eagerly().

    Returns:
      An object with a `forward` method returning a tuple of (forward_function :
      AtomicFunction, augmented_arguments : List), and a corresponding
      `record` method which takes outputs from the forward function and records
      the operation. forward_function should be called with augmented_arguments.
    """
        if executing_eagerly:
            input_tangents = forwardprop_util.pack_tangents(args)
        else:
            input_tangents = forwardprop_util.TangentInfo()
        need_gradients_for_jvps = record.should_record_backprop(input_tangents.tangents)
        cache_key = (need_gradients_for_jvps, input_tangents.indices)
        if possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER:
            if input_tangents.indices or executing_eagerly:
                functions = self._first_order_tape_functions.get(cache_key, None)
                if functions is None:
                    functions = _FirstOrderTapeGradientFunctions(self._func_graph, self._attrs, self._garbage_collector, forwardprop_input_indices=input_tangents.indices, delayed_rewrite_functions=self._delayed_rewrite_functions, need_gradients_for_jvps=need_gradients_for_jvps)
                    self._first_order_tape_functions[cache_key] = functions
                return _ForwardBackwardCall(functions, args, input_tangents.tangents, tape_watching=True)
            else:
                return _ForwardBackwardCall(self._delayed_rewrite_functions, args, input_tangents.tangents, tape_watching=True)
        elif possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER:
            functions = self._higher_order_tape_functions.get(cache_key, None)
            if functions is None:
                functions = _HigherOrderTapeGradientFunctions(self._func_graph, self._attrs, self._garbage_collector, forwardprop_input_indices=input_tangents.indices, delayed_rewrite_functions=self._delayed_rewrite_functions, need_gradients_for_jvps=need_gradients_for_jvps)
                self._higher_order_tape_functions[cache_key] = functions
            return _ForwardBackwardCall(functions, args, input_tangents.tangents, tape_watching=True)
        return _ForwardBackwardCall(self._delayed_rewrite_functions, args, input_tangents.tangents, tape_watching=False)

    @property
    def _as_name_attr_list(self):
        """Returns a `NameAttrList` representing this function."""
        ret = attr_value_pb2.NameAttrList(name=self.name)
        for name, value in self._attrs.items():
            ret.attr[name].CopyFrom(value)
        return ret

    def _structured_signature_summary(self, default_values=False):
        """Returns a string summarizing this function's structured signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
        assert self.function_type is not None
        arg_specs, kwarg_specs = self.structured_input_signature
        arg_names = function_type_utils.to_arg_names(self.function_type)
        arg_names = arg_names[:len(arg_specs)]
        if default_values:
            for i in range(len(arg_names)):
                if not _contains_type_spec(arg_specs[i]):
                    arg_names[i] += '={}'.format(arg_specs[i])
        if kwarg_specs:
            arg_names.append('*')
            for name, spec in kwarg_specs.items():
                arg_names.append(name)
                if default_values and (not _contains_type_spec(spec)):
                    arg_names[-1] += '={}'.format(spec)
        signature = f'{self._func_graph.name}({', '.join(arg_names)})'
        return signature

    def _flat_signature_summary(self):
        """Returns a string summarizing this function's flat signature."""
        assert self._arg_keywords is not None
        assert self._num_positional_args is not None
        arg_names = self._arg_keywords
        if self._num_positional_args > len(arg_names):
            arg_names.extend(('<arg{}>'.format(i + 1) for i in range(len(arg_names), self._num_positional_args)))
        return f'{self._func_graph.name}({', '.join(arg_names)})'

    def pretty_printed_signature(self, verbose=True):
        """Returns a string summarizing the signature of this concrete function."""
        if not verbose:
            return self._structured_signature_summary(default_values=True)

        def pretty_print_spec(spec):
            """Returns a string describing the spec for a single argument."""
            if isinstance(spec, tensor_lib.TensorSpec):
                return '{} Tensor, shape={}'.format(spec.dtype.name, spec.shape)
            elif nest.is_nested(spec):
                pieces = nest.flatten(spec, expand_composites=False)
                markers = [_Marker('<{}>'.format(i + 1)) for i in range(len(pieces))]
                structure = nest.pack_sequence_as(spec, markers)
                result = pprint.pformat(structure, width=10000)
                for marker, piece in zip(markers, pieces):
                    result += '\n      {}: {}'.format(marker, pretty_print_spec(piece))
                return result
            else:
                return repr(spec)
        lines = [self._structured_signature_summary(default_values=True)]
        arg_specs, kwarg_specs = self.structured_input_signature
        names = function_type_utils.to_arg_names(self.function_type)
        arg_details = []
        for name, spec in zip(names[:len(arg_specs)], list(arg_specs)):
            if _contains_type_spec(spec):
                arg_details.append('    {}: {}'.format(name, pretty_print_spec(spec)))
        if kwarg_specs:
            for kwarg in sorted(kwarg_specs):
                spec = kwarg_specs[kwarg]
                if _contains_type_spec(spec):
                    arg_details.append('    {}: {}'.format(kwarg, pretty_print_spec(spec)))
        if arg_details:
            lines.append('  Args:')
            lines.extend(arg_details)
        lines.append('  Returns:')

        def spec_from_value(value):
            if isinstance(value, type_spec.TypeSpec):
                return value
            return type_spec.type_spec_from_value(value)
        lines.append('    {}'.format(pretty_print_spec(nest.map_structure(spec_from_value, self.structured_outputs))))
        return '\n'.join(lines)

    def __repr__(self):
        if self.function_type is not None:
            return '<ConcreteFunction {} at 0x{:X}>'.format(self.pretty_printed_signature(verbose=False), id(self))
        elif not (self._num_positional_args is None or self._arg_keywords is None):
            return '<ConcreteFunction {} at 0x{:X}>'.format(self._flat_signature_summary(), id(self))
        else:
            return object.__repr__(self)

    def __str__(self):
        if self.function_type is not None:
            return 'ConcreteFunction {}'.format(self.pretty_printed_signature())
        else:
            return self.__repr__()

    def _trackable_children(self, save_type='checkpoint', **kwargs):
        """Implements `Trackable`."""
        if save_type == 'checkpoint':
            return {}
        captured_trackables = {}
        for n, (capture, _) in enumerate(self.graph.captures):
            if capture.dtype not in (dtypes.variant, dtypes.resource) and (not resource_variable_ops.is_resource_variable(capture)):
                captured_trackables[f'capture_{n}'] = capture
        return captured_trackables

    def _deserialization_dependencies(self, children):
        return children

    def _export_to_saved_model_graph(self, object_map, tensor_map, **unused_kwargs):
        if not self.graph.saveable:
            raise ValueError(f'Unable to save function {self.name} for the following reason(s):\n' + '\n'.join(self.graph.saving_errors))
        self.add_to_graph()
        object_map[self] = saved_model_exported_concrete.ExportedConcreteFunction(self, tensor_map)
        return []