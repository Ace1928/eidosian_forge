import torch
class GroupNorm(torch.nn.GroupNorm):
    """This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, weight, bias, scale, zero_point, eps=1e-05, affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_groups, num_channels, eps, affine, **factory_kwargs)
        self.weight = weight
        self.bias = bias
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.group_norm(input, self.num_groups, self.weight, self.bias, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedGroupNorm'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_groups, mod.num_channels, mod.weight, mod.bias, float(scale), int(zero_point), mod.eps, mod.affine)
        return new_mod