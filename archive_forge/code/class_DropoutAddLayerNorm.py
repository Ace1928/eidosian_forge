import dropout_layer_norm
import torch
from torch.nn import init
class DropoutAddLayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, prenorm=False, p=0.0, eps=1e-05, residual_in_fp32=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x0, residual=None):
        return dropout_add_layer_norm(x0, residual, self.weight, self.bias, self.p if self.training else 0.0, self.eps, prenorm=self.prenorm, residual_in_fp32=self.residual_in_fp32)