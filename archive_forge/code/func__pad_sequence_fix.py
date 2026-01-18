from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _pad_sequence_fix(attr, kernel_dim=None):
    """Changing onnx's pads sequence to match with mxnet's pad_width
    mxnet: (x1_begin, x1_end, ... , xn_begin, xn_end)
    onnx: (x1_begin, x2_begin, ... , xn_end, xn_end)"""
    new_attr = ()
    if len(attr) % 2 == 0:
        for index in range(int(len(attr) / 2)):
            new_attr = new_attr + attr[index::int(len(attr) / 2)]
        if kernel_dim is not None:
            while len(new_attr) < kernel_dim * 2:
                new_attr = new_attr + (0, 0)
    return new_attr