from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
def PostProcessing(self):
    """Perform postprocessing at the end of gradients().

    We have created the gradient graph at this point. So this function
    can be used to perform any postprocessing on the gradient graph.
    We currently perform the following postprocessing:
      1. Patch the gradient graph if the output of a loop variable
         doesn't depend on its input.
    """
    for _, grad_state in self._map.items():
        for _, b_merge in grad_state.switch_map.items():
            if b_merge.op.inputs[0] == b_merge.op.inputs[1]:
                dtype = b_merge.op.inputs[0].dtype
                shape = b_merge.op.inputs[0].get_shape()
                if shape.is_fully_defined():
                    grad_state.grad_context.Enter()
                    grad_val = constant_op.constant(0, dtype=dtype, shape=shape)
                    next_grad_val = control_flow_ops._NextIteration(grad_val)
                    grad_state.grad_context.Exit()
                else:
                    outer_grad_ctxt = grad_state.grad_context.outer_context
                    if outer_grad_ctxt:
                        outer_grad_ctxt.Enter()
                    enter_grad_op = b_merge.op.inputs[0].op
                    enter_grad = enter_grad_op.inputs[0]
                    grad_shape = array_ops.shape_internal(enter_grad, optimize=False)
                    grad_val = array_ops.zeros(grad_shape)
                    if outer_grad_ctxt:
                        outer_grad_ctxt.Exit()
                    grad_state.grad_context.Enter()
                    next_grad_val = control_flow_ops._NextIteration(grad_val)
                    grad_state.grad_context.Exit()
                b_merge.op._update_input(1, next_grad_val)