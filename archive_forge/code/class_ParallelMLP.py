import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
class ParallelMLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu, process_group: ProcessGroup=None, sequence_parallel=True, bias1=True, bias2=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert ColumnParallelLinear is not None, 'Need to install fused_dense'
        assert RowParallelLinear is not None, 'Need to install fused_dense'
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, process_group, bias=bias1, sequence_parallel=sequence_parallel, **factory_kwargs)
        self.activation = activation
        self.fc2 = RowParallelLinear(hidden_features, out_features, process_group, bias=bias2, sequence_parallel=sequence_parallel, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y