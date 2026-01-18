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
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(self, input_size: int, output_size: int, bias: bool=True, gather_output: bool=False, skip_bias_add: bool=False, params_dtype: Optional[torch.dtype]=None, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        tp_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, tp_size)
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if linear_method is None:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(self.input_size, self.output_size_per_partition, self.input_size, self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
                set_weight_attrs(weight, {'weight_loader': self.weight_loader})
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            set_weight_attrs(self.bias, {'output_dim': 0, 'weight_loader': self.weight_loader})
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        output_dim = getattr(param, 'output_dim', None)
        param_data = param.data
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = self.linear_method.apply_weights(self.linear_weights, input_, bias)
        if self.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return (output, output_bias)