from functools import partial
from typing import Optional
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup
from flash_attn.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from flash_attn.utils.distributed import (
class ParallelFusedMLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation='gelu_approx', process_group: ProcessGroup=None, bias1=True, bias2=True, sequence_parallel=True, checkpoint_lvl=0, heuristic='auto', device=None, dtype=None):
        """
        process_group is required. We're doing Tensor Parallel with sequence parallelism:
        we do an all_gather of x before doing the matmul, gelu, then matmul.
        Finally we do a reduce_scatter of the output.

        checkpoint_lvl (increasing lvl means slower but more memory saving):
            0: no recomputation in the bwd
            1: recompute gelu_out in the bwd
            2: recompute pre_act and gelu_out in the bwd
        heuristic:
            -1: don't fuse gemm + gelu (separate kernel)
            0..4: use this heuristic for the algo section in the fused gemm + gelu
            'auto': heuristic will be picked automatically:
                For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
        """
        assert checkpoint_lvl in [0, 1, 2]
        assert activation in ['gelu_approx', 'relu', 'sqrelu']
        assert process_group is not None
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.activation = activation
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.checkpoint_lvl = checkpoint_lvl
        self.heuristic = heuristic if activation != 'sqrelu' else -1
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, process_group, bias=bias1, **factory_kwargs)
        self.fc2 = RowParallelLinear(hidden_features, out_features, process_group, bias=bias2, **factory_kwargs)

    def forward(self, x):
        dtype = x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
        if self.heuristic == 'auto':
            if self.activation == 'gelu_approx':
                cuda_ver = tuple(map(int, torch.version.cuda.split('.')))
                heuristic = 0 if cuda_ver >= (11, 8) else 1 if dtype == torch.float16 else -1
            else:
                heuristic = 0
        else:
            heuristic = self.heuristic
        out = fused_mlp_func(x, self.fc1.weight, self.fc2.weight, self.fc1.bias, self.fc2.bias, activation=self.activation, save_pre_act=self.training, checkpoint_lvl=self.checkpoint_lvl, heuristic=heuristic, process_group=self.process_group, sequence_parallel=self.sequence_parallel)
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)