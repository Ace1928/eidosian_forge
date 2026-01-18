import torch
class MkldnnConv1d(_MkldnnConvNd):

    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]