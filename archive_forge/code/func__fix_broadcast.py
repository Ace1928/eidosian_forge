from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _fix_broadcast(op_name, inputs, broadcast_axis, proto_obj):
    """A workaround to reshape bias term to (1, num_channel)."""
    if int(len(proto_obj._params)) > 0:
        assert len(list(inputs)) == 2
        input0_shape = get_input_shape(inputs[0], proto_obj)
        reshape_shape = list(len(input0_shape) * (1,))
        reshape_shape[broadcast_axis] = -1
        reshape_shape = tuple(reshape_shape)
        reshape_op_sym = symbol.reshape(inputs[1], shape=reshape_shape)
        op_sym = getattr(symbol, op_name)(inputs[0], reshape_op_sym)
    else:
        op_sym = op_name
    return op_sym