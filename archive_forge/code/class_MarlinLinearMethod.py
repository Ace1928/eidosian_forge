from typing import Any, Dict, List, Optional
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
class MarlinLinearMethod(LinearMethodBase):
    """Linear method for Marlin.

    Args:
        quant_config: The Marlin quantization config.
    """

    def __init__(self, quant_config: MarlinConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int, output_size_per_partition: int, input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        del output_size
        if params_dtype != torch.float16:
            raise ValueError(f'The params dtype must be float16, but got {params_dtype}')
        if output_size_per_partition % self.quant_config.min_n_threads != 0:
            raise ValueError(f'Weight output_size_per_partition = {output_size_per_partition} is not divisible by min_n_threads = {self.quant_config.min_n_threads}.')
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(f'Weight output_size_per_partition = {output_size_per_partition} is not divisible by pack_factor = {self.quant_config.pack_factor}.')
        if input_size_per_partition % self.quant_config.min_k_threads != 0:
            raise ValueError(f'Weight input_size_per_partition = {input_size_per_partition} is not divisible by min_k_threads = {self.quant_config.min_k_threads}.')
        if self.quant_config.group_size != -1 and input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(f'Weight input_size_per_partition = f{input_size_per_partition} is not divisible by group_size = {self.quant_config.group_size}.')
        num_tiles_per_perm = self.quant_config.perm_len // self.quant_config.tile_size ** 2
        if output_size_per_partition % num_tiles_per_perm != 0:
            raise ValueError('Each permutation group must reside on the same gpu')
        qweight = Parameter(torch.empty(input_size_per_partition // self.quant_config.tile_size, output_size_per_partition * self.quant_config.tile_size // self.quant_config.pack_factor, device='cuda', dtype=torch.int32), requires_grad=False)
        set_weight_attrs(qweight, {'input_dim': 0, 'output_dim': 1, 'packed_dim': 1, 'pack_factor': self.quant_config.pack_factor, 'marlin_tile_size': self.quant_config.tile_size})
        input_groups = 1 if self.quant_config.group_size == -1 else input_size_per_partition // self.quant_config.group_size
        scales = Parameter(torch.empty(input_groups, output_size_per_partition, device='cuda', dtype=params_dtype), requires_grad=False)
        set_weight_attrs(scales, {'input_dim': None if input_groups == 1 else 0, 'output_dim': 1})
        max_workspace_size = output_size_per_partition // self.quant_config.min_n_threads * self.quant_config.max_parallel
        workspace = Parameter(torch.zeros(max_workspace_size, device='cuda', dtype=torch.int), requires_grad=False)
        return {'B': qweight, 's': scales, 'workspace': workspace}

    def apply_weights(self, weights: Dict[str, Any], x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
        qweight = weights['B']
        scales = weights['s']
        workspace = weights['workspace']
        x_2d = x.view(-1, x.shape[-1])
        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = scales.shape[1]
        output_2d = ops.marlin_gemm(x_2d, qweight, scales, workspace, size_m, size_n, size_k)
        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1],))
        if bias is not None:
            output.add_(bias)
        return output