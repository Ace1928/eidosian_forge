import torch
class _MkldnnConvNd(torch.jit.ScriptModule):
    """Common base of MkldnnConv1d and MkldnnConv2d."""
    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module):
        super().__init__()
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            self.register_buffer('bias', torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def forward(self, x):
        return torch.mkldnn_convolution(x, self.weight, self.bias, self.padding, self.stride, self.dilation, self.groups)