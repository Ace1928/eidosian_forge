import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from flash_attn.ops.activations import sqrelu_bwd, sqrelu_fwd
from flash_attn.ops.triton.linear import triton_dgrad_act, triton_linear_act
class FusedDenseSqreluDense(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, bias1=True, bias2=True, checkpoint_lvl=0, device=None, dtype=None):
        """
        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute gelu_in and gelu_out in the bwd
        """
        assert checkpoint_lvl in [0, 1, 2]
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        assert bias1 == True, 'DenseSqreluDense module without bias is currently not supported'
        assert bias2 == True, 'DenseSqreluDense module without bias is currently not supported'
        self.checkpoint_lvl = checkpoint_lvl
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x):
        assert x.is_cuda
        return fused_dense_sqrelu_dense_function(x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.checkpoint_lvl)