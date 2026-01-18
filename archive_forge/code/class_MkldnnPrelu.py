import torch
class MkldnnPrelu(torch.jit.ScriptModule):

    def __init__(self, dense_module, dtype):
        super().__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.training = state[1]

    @torch.jit.script_method
    def forward(self, x):
        x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        y_mkldnn = torch.prelu(x_mkldnn, self.weight)
        y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        return y