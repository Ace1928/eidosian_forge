import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.punica import add_lora, add_lora_slice, bgmv
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import split_tensor_along_last_dim
class RowParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_lora_weights(self, max_loras: int, lora_config: LoRAConfig, model_config: Optional[PretrainedConfig]=None) -> None:
        self.lora_a_stacked = torch.zeros((max_loras, 1, lora_config.max_lora_rank, self.base_layer.weight.shape[1]), dtype=lora_config.lora_dtype, device=self.base_layer.weight.device)
        self.lora_b_stacked = torch.zeros((max_loras, 1, self.base_layer.weight.shape[0], lora_config.max_lora_rank), dtype=lora_config.lora_dtype, device=self.base_layer.weight.device)
        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor, embeddings_tensor: Optional[torch.Tensor]):
        self.reset_lora(index)
        if self.base_layer.tp_size > 1:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.base_layer.weight.shape[1]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            lora_a = lora_a[start_idx:end_idx, :]
        self.lora_a_stacked[index, 0, :lora_a.shape[1], :lora_a.shape[0]].copy_(lora_a.T, non_blocking=True)
        self.lora_b_stacked[index, 0, :lora_b.shape[1], :lora_b.shape[0]].copy_(lora_b.T, non_blocking=True)

    def set_mapping(self, base_indices: torch.Tensor, sampler_indices: torch.Tensor, sampler_indices_padded: torch.Tensor, embeddings_indices: torch.Tensor, indices_len: List[int]):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(self.base_layer.linear_weights, x)
        _apply_lora(x, self.lora_a_stacked, self.lora_b_stacked, self.indices[:self.indices_len[0]], output)
        return output

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.apply_weights(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
        if not self.base_layer.skip_bias_add:
            output = output_ + self.base_layer.bias if self.base_layer.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return (output, output_bias)

    @property
    def weight(self):
        return self.base_layer.weight