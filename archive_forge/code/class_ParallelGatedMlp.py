import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
class ParallelGatedMlp(nn.Module):
    """Parallel GatedMlp"""

    def __init__(self, in_features, process_group, hidden_features=None, out_features=None, activation=F.sigmoid, bias1=True, bias2=True, multiple_of=128, sequence_parallel=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * in_features / 3)
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError('fused_dense is not installed')
        self.fc1 = ColumnParallelLinear(in_features, 2 * hidden_features, process_group, bias=bias1, sequence_parallel=sequence_parallel, **factory_kwargs)
        self.activation = activation
        self.fc2 = RowParallelLinear(hidden_features, out_features, process_group, bias=bias2, sequence_parallel=sequence_parallel, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        if self.activation == F.sigmoid:
            y = F.glu(y, dim=-1)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y