import numpy as np
from onnx.reference.op_run import OpRun
class DeformConv(OpRun):

    def _run(self, X, W, offset, B=None, mask=None, dilations=None, group=None, kernel_shape=None, offset_group=None, pads=None, strides=None):
        if len(X.shape) < 3:
            raise ValueError(f'X must have at least 3 dimensions but its shape is {X.shape}.')
        return (_deform_conv_implementation(X, W, offset, B, mask, dilations, group, kernel_shape, offset_group, pads, strides),)