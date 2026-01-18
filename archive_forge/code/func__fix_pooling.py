from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _fix_pooling(pool_type, inputs, new_attr):
    """onnx pooling operator supports asymmetrical padding
    Adding pad operator before pooling in mxnet to work with onnx"""
    stride = new_attr.get('stride')
    kernel = new_attr.get('kernel')
    padding = new_attr.get('pad')
    p_value = new_attr.get('p_value')
    if stride is None:
        stride = (1,) * len(kernel)
    if padding is None:
        padding = (0,) * len(kernel) * 2
    if len(kernel) == 1:
        dummy_axis = 2
        padding = (0, padding[0], 0, padding[1])
        pad_width = (0, 0, 0, 0) + _pad_sequence_fix(padding, kernel_dim=2)
        curr_sym = symbol.expand_dims(inputs[0], axis=dummy_axis)
        new_pad_op = symbol.pad(curr_sym, mode='edge', pad_width=pad_width)
        new_pad_op = symbol.split(new_pad_op, axis=dummy_axis, num_outputs=1, squeeze_axis=1)
    else:
        pad_width = (0, 0, 0, 0) + _pad_sequence_fix(padding, kernel_dim=len(kernel))
        curr_sym = inputs[0]
        if pool_type == 'max':
            new_pad_op = symbol.pad(curr_sym, mode='edge', pad_width=pad_width)
        else:
            new_pad_op = symbol.pad(curr_sym, mode='constant', pad_width=pad_width)
    if pool_type == 'lp':
        new_pooling_op = symbol.Pooling(new_pad_op, pool_type=pool_type, stride=stride, kernel=kernel, p_value=p_value)
    else:
        new_pooling_op = symbol.Pooling(new_pad_op, pool_type=pool_type, stride=stride, kernel=kernel)
    return new_pooling_op