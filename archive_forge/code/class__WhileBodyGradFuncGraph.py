import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
class _WhileBodyGradFuncGraph(util.WhileBodyFuncGraph):
    """FuncGraph for the gradient function of the body of a While op.

  Contains the logic for capturing the tensors from the body of the forward
  While op which is as follows:
  1. If the tensor is of resource type (these are not accumulated):
     a. Ensure that the tensor is a loop invariant, i.e., it exists in both loop
        inputs and outputs at the same index.
     b. Lookup the corresponding resource tensor in the forward outer graph and
        try to capture that.
  2. If the tensor is not of resource type:
     a. Create an accumulator for that tensor and output it from the forward
        pass. Note this also requires adding it as an input to the forward pass.
     b. Capture the accumulator from the forward pass in this FuncGraph. This
        will later be resolved to the correct output of the forward While op.
     c. Pop a value from the captured placeholder and use it as the captured
        value for the forward pass tensor.

  This only allows capturing tensors in the forward graph. A ValueError is
  raised if an attempt is made to capture a tensor not in the forward graph.
  To manually capture a tensor that is not in the forward graph, call `capture`
  with `allowlisted=True`.

  Note: The `captures` dict does not contain the forward tensor since it is not
  directly captured. It contains the accumulator corresponding to this forward
  tensor.

  Attributes:
    while_op_needs_rewrite: True if any non-resource intermediates were
      captured, meaning the forward While op needs to be rewritten to output the
      corresponding accumulators.
    extra_inputs: list of EmptyTensorList tensors to be used as initial input to
    the new accumulators in the forward graph. It may also contain external
    captures of the custom gradient function.
    internal_capture_to_output: dict from a tensor_id(captured placeholder) to
      the corresponding tensor that needs to be added to the list of outputs.
      For instance, when capturing an accumulator TensorList this contains the
      TensorList obtained after popping a tensor from the list. Other entries
      in this dict are expected, though not enforced, to be identities.
      This dict is needed because these output tensors need to be added to
      FuncGraph.outputs "after" the tensors returned from the gradient function.
  """

    def __init__(self, name, forward_cond_graph, forward_body_graph, maximum_iterations, forward_while_op, body_graph_inputs, body_graph_outputs):
        super(_WhileBodyGradFuncGraph, self).__init__(name)
        self.extra_inputs = []
        self.internal_capture_to_output = {}
        self._forward_graph = forward_body_graph
        self._forward_cond_graph = forward_cond_graph
        self._maximum_iterations = maximum_iterations
        self._forward_while_op = forward_while_op
        self._indirect_captures = {}

    @property
    def while_op_needs_rewrite(self):
        return self.extra_inputs

    def _create_op_internal(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        optimized_reduction_ops = {'Shape', 'Size', 'Rank', 'TensorListElementShape', 'TensorListLength'}
        if op_type in optimized_reduction_ops and (not util.output_all_intermediates()) and all((input.graph is self._forward_graph for input in inputs)) and all((_get_accumulator(input) is None for input in inputs)) and (not util_v1.GraphOrParentsInXlaContext(self._forward_graph)) and (not util.graph_wrapped_for_higher_order_tape_gradients(self._forward_graph)):
            return self._move_op_to_forward_graph(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)
        return super(_WhileBodyGradFuncGraph, self)._create_op_internal(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)

    def _move_op_to_forward_graph(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        if not hasattr(self._forward_graph, '_optimized_reduction_ops_cache'):
            self._forward_graph._optimized_reduction_ops_cache = {}
        cache_key = self._get_optimized_reduction_ops_cache_key(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
        cached_op = self._forward_graph._optimized_reduction_ops_cache.get(cache_key)
        if cached_op is not None:
            return cached_op
        with self._forward_graph.as_default():
            name = ops.name_from_scope_name(name)
            result = self._forward_graph._create_op_internal(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)
            self._forward_graph._optimized_reduction_ops_cache[cache_key] = result
            return result

    def _get_optimized_reduction_ops_cache_key(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        inputs = tuple(map(lambda t: t.ref(), inputs))
        if dtypes is not None:
            dtypes = tuple(dtypes)
        if input_types is not None:
            input_types = tuple(input_types)
        if attrs is not None:
            hashable_attrs = []
            for attr_name, attr_value in sorted(attrs.items()):
                hashable_attrs.append((attr_name, attr_value.SerializeToString()))
            attrs = tuple(hashable_attrs)
        if op_def is not None:
            op_def = op_def.SerializeToString()
        return OptimizedReductionOpsCacheKey(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)

    def _capture_helper(self, tensor, name):
        """Implements the capturing described in the class docstring."""
        captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
        if captured_tensor is not None:
            return captured_tensor
        if tensor.graph is not self._forward_graph:
            already_captured = id(tensor) in self.function_captures.by_val_internal
            captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)
            if not already_captured:
                self.internal_capture_to_output[ops.tensor_id(captured_tensor)] = captured_tensor
                self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
            return captured_tensor
        while tensor.op.type == 'Identity':
            tensor = tensor.op.inputs[0]
        captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
        if captured_tensor is not None:
            return captured_tensor
        if _is_loop_invariant(tensor, self._forward_graph.inputs, self._forward_graph.outputs):
            captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)
            self.internal_capture_to_output[ops.tensor_id(captured_tensor)] = captured_tensor
            self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
            return captured_tensor
        if constant_op.is_constant(tensor):
            real_value = constant_op.constant(tensor_util.constant_value(tensor), dtype=tensor.dtype)
            self._indirect_captures[ops.tensor_id(tensor)] = real_value
            return real_value
        if tensor.dtype == dtypes.resource:
            return self._resource_capture_helper(tensor)
        accumulator = _get_accumulator(tensor)
        if accumulator is None:
            with self._forward_graph.outer_graph.as_default():
                with util.clear_control_inputs():
                    tensor_list = list_ops.empty_tensor_list(element_dtype=tensor.dtype, element_shape=tensor.shape, max_num_elements=self._maximum_iterations, name=_build_accumulator_name(tensor))
            self.extra_inputs.append(tensor_list)
            with self._forward_graph.as_default():
                accumulator = list_ops.tensor_list_push_back(tensor_list, tensor)
            self._forward_graph.outputs.append(accumulator)
            with self._forward_cond_graph.as_default():
                self._forward_cond_graph.capture(tensor_list)
        captured_accumulator = super(_WhileBodyGradFuncGraph, self)._capture_helper(accumulator, name)
        new_tensor_list, captured_tensor = list_ops.tensor_list_pop_back(captured_accumulator, element_dtype=tensor.dtype)
        self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
        self.internal_capture_to_output[ops.tensor_id(captured_accumulator)] = new_tensor_list
        return captured_tensor

    def _resource_capture_helper(self, tensor):
        """Returns the captured resource tensor.

    Resource-type tensors are not accumulated. If a resource tensor exists in
    the loop body it must either be a loop input or an output of a nested While
    op inside the loop body which had captured the external resource.

    Args:
      tensor: the external resource Tensor to be captured.

    Returns:
      Tensor in this graph.
    """
        assert tensor.dtype == dtypes.resource
        forward_graph_input_names = [t.name for t in self._forward_graph.inputs]
        forward_graph_name_to_opdef = {op.name: op.node_def for op in self._forward_graph.get_operations()}
        index = util.resource_input_index(tensor.name, forward_graph_input_names, forward_graph_name_to_opdef, self._forward_graph._functions)
        input_placeholder = self._forward_graph.inputs[index]
        tensor_in_outer_graph = self._forward_graph._while.inputs[index]
        assert input_placeholder.dtype == dtypes.resource
        assert tensor_in_outer_graph.dtype == dtypes.resource
        if index != util.resource_input_index(self._forward_graph.outputs[index].name, forward_graph_input_names, forward_graph_name_to_opdef, self._forward_graph._functions):
            raise AssertionError(f'Resource tensors must be loop invariants {tensor_in_outer_graph}')
        self._indirect_captures[ops.tensor_id(tensor)] = self.capture(tensor_in_outer_graph)
        return self._indirect_captures[ops.tensor_id(tensor)]