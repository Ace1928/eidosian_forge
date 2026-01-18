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
class _DelayedRewriteGradientFunctions(object):
    """Caches forward/backward functions with a delayed forward rewrite."""

    def __init__(self, atomic_fn, func_graph_deleter):
        """Construct an inference function and initialize caches."""
        self._cached_function_pairs = {}
        self._func_graph = atomic_fn.graph
        self._inference_function = atomic_fn
        self._attrs = atomic_fn.attributes
        self._gradient_name = None
        self._num_inference_outputs = len(self._func_graph.outputs)
        self._func_graph_deleter = func_graph_deleter

    def forward_backward(self, num_doutputs=None):
        """A possibly-cached pair of forward and backward functions."""
        if num_doutputs is None:
            num_doutputs = self._num_inference_outputs
        forward_backward = self._cached_function_pairs.get(num_doutputs)
        if forward_backward is not None:
            return forward_backward
        forward, backward = self._construct_forward_backward(num_doutputs)
        self._cached_function_pairs[num_doutputs] = (forward, backward)
        return (forward, backward)

    def _construct_forward_backward(self, num_doutputs):
        """Constructs a pair of forward and backward functions.

    Args:
      num_doutputs: The constructed backprop function will take output gradients
        for the first `num_doutputs` outputs of the forward function. Defaults
        to the number of outputs for the inference function, but when
        higher-order gradients are computed this will increase to include side
        outputs.

    Returns:
      A pair of (forward_function, backward_function):
        forward_function: A re-generated inference function (an
          AtomicFunction) to account for new side outputs, if any extra
          were required when building the backward pass.
        backward_function: A ConcreteFunction that Takes `num_doutputs`
          arguments and returns gradients with respect to inputs of the forward
          function.
    """
        trainable_outputs = [output for output in self._func_graph.outputs[:num_doutputs] if backprop_util.IsTrainable(output)]
        signature = []
        for t in trainable_outputs:
            signature.append(tensor_lib.TensorSpec(*default_gradient.shape_and_dtype(t)))

        def _backprop_function(*grad_ys):
            with ops.device(None):
                return gradients_util._GradientsHelper(trainable_outputs, self._func_graph.inputs, grad_ys=grad_ys, src_graph=self._func_graph)
        with self._func_graph.as_default():
            backwards_graph = func_graph_module.FuncGraph(_backward_name(self._func_graph.name))
            func_graph_module.func_graph_from_py_func(name=backwards_graph.name, python_func=_backprop_function, args=[], kwargs={}, signature=signature, func_graph=backwards_graph)
            backwards_graph_captures = backwards_graph.external_captures
            captures_from_forward = [c for c in backwards_graph_captures if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]
            existing_outputs = object_identity.ObjectIdentitySet(self._func_graph.outputs)
            for capture in captures_from_forward:
                if capture not in existing_outputs:
                    existing_outputs.add(capture)
                    self._func_graph.outputs.append(capture)
            forward_function, backward_function = _create_forward_backward_with_graph(self._attrs, self._func_graph, backwards_graph)
            return (forward_function, backward_function)

    def _rewrite_forward_and_call_backward(self, op, *doutputs):
        """Add outputs to the forward call and feed them to the grad function."""
        forward_function, backwards_function = self.forward_backward(len(doutputs))
        if not backwards_function.outputs:
            return backwards_function.structured_outputs
        op.graph._add_function_recursive(forward_function)
        op._set_func_attr('f', forward_function.name)
        op._set_type_list_attr('Tout', [o.dtype.as_datatype_enum for o in forward_function.function_type.flat_outputs])
        truncated_outputs = forward_function.function_type.flat_outputs[len(op.outputs):]
        op._add_outputs([o.dtype.as_datatype_enum for o in truncated_outputs], [o.shape for o in truncated_outputs])
        for i in range(len(op.outputs)):
            output_type = forward_function.function_type.flat_outputs[i]
            handle_data = output_type.dtype._handle_data
            if handle_data:
                handle_data_util.set_handle_data(op.outputs[i], handle_data.shape_inference)
        capture_mapping = dict(zip((ops.tensor_id(t) for t in self._func_graph.outputs), op.outputs))
        remapped_captures = [capture_mapping.get(ops.tensor_id(capture), capture) for capture in backwards_function.captured_inputs]
        cleaned_doutputs = []
        for doutput, placeholder in zip(doutputs, self._func_graph.outputs):
            if backprop_util.IsTrainable(placeholder):
                if isinstance(doutput, indexed_slices.IndexedSlices):
                    cleaned_doutputs.append(ops.convert_to_tensor(doutput))
                elif doutput is not None:
                    cleaned_doutputs.append(doutput)
                else:
                    cleaned_doutputs.append(default_gradient.zeros_like(placeholder))
        return backwards_function._call_flat(cleaned_doutputs, remapped_captures)

    def get_gradient_function(self):
        """Returns gradient function.

    The gradient rewrites an inference call op to a forward call op, but does
    not modify a pre-existing forward call op. It then computes the gradient
    from the output's gradients and the side outputs of the forward op.
    """
        return self._rewrite_forward_and_call_backward

    def forward(self, inference_args=None, input_tangents=None):
        """A forward function with only user-specified outputs.

    The call operation for the returned inference function can be rewritten into
    a forward function. This only happens if the backward function (from the
    `backward` method) ends up being used to compute gradients.

    This approach avoids constructing unnecessary graphs, but it only works if
    we are calling this function when not executing eagerly.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function. Unused, but taken for compatibility with
        _TapeGradientFunctions.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`. Unused; if required, tape functions must be used
        instead.

    Returns:
      An atomic_function.AtomicFunction.
    """
        del inference_args
        if input_tangents:
            raise errors.InternalError('unexpectedly got forwardprop information in a class that does not support forwardprop.')
        return self._inference_function

    def _backward(self, outputs):
        """Fetch a backward function for `outputs` from the forward function."""

        def _backward_function(*args):
            call_op = outputs[0].op
            return self._rewrite_forward_and_call_backward(call_op, *args)
        return (_backward_function, outputs)

    def record(self, flat_outputs, inference_args, input_tangents):
        """Record the function call operation.

    _DelayedRewriteGradientFunctions supports only first-order backprop tape
    gradients (and then only when graph building). It does not work with
    higher-order tape gradients or forward autodiff, but does work with
    higher-order symbolic gradients (tf.gradients).

    Args:
      flat_outputs: The result of running `forward`.
      inference_args: A flat list of Tensors with inference inputs to the
        operation.
      input_tangents: A flat list of Tensors with input tangents consumed by the
        operation.
    """
        backward_function, to_record = self._backward(flat_outputs)
        record.record_operation(self._inference_function.cached_definition.signature.name, to_record, inference_args + input_tangents, backward_function)