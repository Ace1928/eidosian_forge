import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _compile_internal(computation, inputs=None):
    """Builds graph operators that compiles and symbolically executes computation.

  Args:
    computation: A Python function that builds the computation to compile and
      execute.
    inputs: A list of inputs or `None` (equivalent to an empty list). Each input
      can be a nested structure containing values that are convertible to
      tensors. Note that passing an N-dimension list of compatible values will
      result in a N-dimension list of scalar tensors rather than a single Rank-N
      tensors. If you need different behavior, convert part of inputs to tensors
      with `tf.convert_to_tensor`.

  Returns:
    Same data structure as if computation(*inputs) is called directly with some
    exceptions for correctness. Exceptions include: 1) None output 2) Single
    value output 3) Operation-only outputs
  Raises:
    ValueError: If any element in computation outputs is neither an operations
      or a value that can be converted to tensor.
    ValueError: If computation outputs is non-flat and contains any Operations.
    TypeError: If `inputs` is not a list or tuple.
  """
    if inputs is None:
        inputs = []
    if not isinstance(inputs, collections_abc.Sequence):
        raise TypeError('inputs must be a list')
    flat_inputs = nest.flatten(inputs)
    flat_inputs = [ops.convert_to_tensor(x) for x in flat_inputs]
    cluster_name = ops.get_default_graph().unique_name('cluster')
    pivot = control_flow_ops.no_op(name=cluster_name + '/pivot')
    context = XLACompileContext(name=cluster_name, pivot=pivot)
    try:
        context.Enter()
        flat_inputs = [array_ops.identity(x, name='input_{}'.format(i)) for i, x in enumerate(flat_inputs)]
        computation_inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_inputs)
        vscope = variable_scope.get_variable_scope()
        saved_use_resource = vscope.use_resource
        vscope.set_use_resource(True)
        with _disable_summary_context():
            outputs = computation(*computation_inputs)
        vscope.set_use_resource(saved_use_resource)
        outputs_is_flat = is_flat(outputs)
        if outputs_is_flat:
            output_tensors, control_deps = _postprocess_flat_outputs(outputs)
        else:
            output_tensors, control_deps = _postprocess_non_flat_outputs(outputs)
        context.ExitResult(output_tensors)
    finally:
        context.report_unsupported_operations()
        context.Exit()
    if not output_tensors:
        return control_flow_ops.group(control_deps, name='output_0')
    output_tensors = [xla_ops.xla_cluster_output(o, name='output{}'.format(i)) for i, o in enumerate(output_tensors)]
    with ops.control_dependencies(control_deps):
        output_tensors = [array_ops.identity(o, name='output_%d' % i) for i, o in enumerate(output_tensors)]
    if not outputs_is_flat:
        output_tensors = nest.pack_sequence_as(structure=outputs, flat_sequence=output_tensors)
    return output_tensors