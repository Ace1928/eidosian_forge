from torch import nn
class DeQuantStub(nn.Module):
    """Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig=None):
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x