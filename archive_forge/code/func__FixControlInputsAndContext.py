import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _FixControlInputsAndContext(self, enters):
    graph = ops.get_default_graph()
    for e in enters:
        if isinstance(e, tensor_lib.Tensor):
            xs = [e]
        else:
            raise TypeError(f"'enters' must be a list of Tensors. Received: {type(e)}.")
        for x in xs:
            inp_op = x.op.inputs[0].op
            control_inputs = graph._control_dependencies_for_inputs([inp_op])
            outer_control_inputs = []
            for op in control_inputs:
                keep_as_control_input = True
                op_ctxt = util.GetOutputContext(op)
                outer_ctxt = self.outer_context
                outer_while_context = None if outer_ctxt is None else outer_ctxt.GetWhileContext()
                while outer_ctxt != op_ctxt:
                    if outer_ctxt is None or outer_ctxt == outer_while_context:
                        keep_as_control_input = False
                        break
                    outer_ctxt = outer_ctxt.outer_context
                if keep_as_control_input:
                    outer_control_inputs.append(op)
            x.op._set_control_flow_context(self)
            x.op._add_control_inputs(outer_control_inputs)
            graph._record_op_seen_by_control_dependencies(x.op)