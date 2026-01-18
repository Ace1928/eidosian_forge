from onnx.reference.ops.op_pool_common import CommonPool
class AveragePool_11(CommonPool):

    def _run(self, x, auto_pad=None, ceil_mode=None, kernel_shape=None, pads=None, strides=None, count_include_pad=None):
        return CommonPool._run(self, 'AVG', count_include_pad, x, auto_pad=auto_pad, ceil_mode=ceil_mode, dilations=None, kernel_shape=kernel_shape, pads=pads, strides=strides)