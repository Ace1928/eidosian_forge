import enum
from enum import Enum
from typing import Any, Dict, List, Optional
from fractions import Fraction
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.quantization.base_config import (
class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int, output_size_per_partition: int, input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        del output_size
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError('The input size is not aligned with the quantized weight shape. This can be caused by too large tensor parallel size.')
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError('The output size is not aligned with the quantized weight shape. This can be caused by too large tensor parallel size.')
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if input_size != input_size_per_partition and self.quant_config.group_size != -1:
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0
        qweight = Parameter(torch.empty(input_size_per_partition // self.quant_config.pack_factor, output_size_per_partition, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(qweight, {'input_dim': 0, 'output_dim': 1, 'packed_dim': 0, 'pack_factor': self.quant_config.pack_factor})
        g_idx = Parameter(torch.tensor([i // self.quant_config.group_size for i in range(input_size_per_partition)], dtype=torch.int32), requires_grad=False)
        set_weight_attrs(g_idx, {'input_dim': 0, 'ignore_warning': True})
        qzeros = Parameter(torch.empty(scale_and_zero_size, output_size_per_partition // self.quant_config.pack_factor, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(qzeros, {'input_dim': scale_and_zero_input_dim, 'output_dim': 1, 'packed_dim': 1, 'pack_factor': self.quant_config.pack_factor})
        scales = Parameter(torch.empty(scale_and_zero_size, output_size_per_partition, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(scales, {'input_dim': scale_and_zero_input_dim, 'output_dim': 1})
        return {'qweight': qweight, 'g_idx': g_idx, 'qzeros': qzeros, 'scales': scales, 'exllama_state': exllama_state}

    def apply_weights(self, weights: Dict[str, Any], x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
        qweight = weights['qweight']
        out_shape = x.shape[:-1] + (qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])
        if weights['exllama_state'] == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                weights['g_idx'] = torch.argsort(weights['g_idx']).to(torch.int)
            else:
                weights['g_idx'] = torch.empty((1, 1), device='meta')
            weights['exllama_state'] = ExllamaState.READY
            ops.gptq_shuffle(weights['qweight'], weights['g_idx'], self.quant_config.weight_bits)
        output = ops.gptq_gemm(reshaped_x, weights['qweight'], weights['qzeros'], weights['scales'], weights['g_idx'], weights['exllama_state'] == ExllamaState.READY, self.quant_config.weight_bits)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)