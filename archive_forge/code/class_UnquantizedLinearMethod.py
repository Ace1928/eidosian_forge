from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.utils import (
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization.

    Args:
        separate_bias_add: If true, add bias separately after matrix
                           multiplication.
    """

    def __init__(self, separate_bias_add: bool=False):
        self.separate_bias_add = separate_bias_add

    def create_weights(self, input_size_per_partition: int, output_size_per_partition: int, input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        weight = Parameter(torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(weight, {'input_dim': 1, 'output_dim': 0})
        return {'weight': weight}

    def apply_weights(self, weights: Dict[str, torch.Tensor], x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
        weight = weights['weight']
        if self.separate_bias_add:
            if bias:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
        return F.linear(x, weight, bias)