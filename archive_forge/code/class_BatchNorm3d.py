import torch
import torch.ao.nn.intrinsic as nni
class BatchNorm3d(_BatchNorm):
    """This is the quantized version of :class:`~torch.nn.BatchNorm3d`.
    """
    _NNI_BN_RELU_MODULE = nni.BNReLU3d

    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedBatchNorm3d'

    def _check_input_dim(self, input):
        if len(input.shape) != 5:
            raise ValueError('Input shape must be `(N, C, H, W)`!')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.batch_norm3d(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod):
        return _BatchNorm.from_float(cls, mod)