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
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(self, input_size: int, output_size: int, bias: bool=True, input_is_parallel: bool=True, skip_bias_add: bool=False, params_dtype: Optional[torch.dtype]=None, reduce_results: bool=True, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.skip_bias_add = skip_bias_add
        if linear_method is None:
            linear_method = UnquantizedLinearMethod()
        self.linear_method = linear_method
        self.linear_weights = self.linear_method.create_weights(self.input_size_per_partition, self.output_size, self.input_size, self.output_size, self.params_dtype)
        for name, weight in self.linear_weights.items():
            if isinstance(weight, torch.Tensor):
                self.register_parameter(name, weight)
                set_weight_attrs(weight, {'weight_loader': self.weight_loader})
        if not reduce_results and (bias and (not skip_bias_add)):
            raise ValueError('When not reduce the results, adding bias to the results can lead to incorrect results')
        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {'output_dim': 0, 'weight_loader': self.weight_loader})
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        input_dim = getattr(param, 'input_dim', None)
        param_data = param.data
        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.linear_method.apply_weights(self.linear_weights, input_parallel)
        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return (output, output_bias)