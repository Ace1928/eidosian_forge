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
class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def create_lora_weights(self, max_loras: int, lora_config: LoRAConfig, model_config: Optional[PretrainedConfig]=None) -> None:
        n_slices = 2
        if not (len(self.base_layer.output_sizes) == n_slices and self.base_layer.output_sizes[0] == self.base_layer.output_sizes[1]):
            raise ValueError('LoRAColumnParallelLinear2Slice requires 2 slices with the same size.')
        self.tp_size = get_tensor_model_parallel_world_size()
        self.lora_a_stacked = tuple((torch.zeros(max_loras, 1, lora_config.max_lora_rank, self.base_layer.weight.shape[1], dtype=lora_config.lora_dtype, device=self.base_layer.weight.device) for _ in range(n_slices)))
        self.lora_b_stacked = tuple((torch.zeros(max_loras, 1, self.base_layer.weight.shape[0] // 2, lora_config.max_lora_rank, dtype=lora_config.lora_dtype, device=self.base_layer.weight.device) for _ in range(n_slices)))
        self.indices: Optional[torch.Tensor] = None
        self.output_dim = self.lora_b_stacked[0].shape[2]

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_b_stacked[1][index] = 0

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor, embeddings_tensor: Optional[torch.Tensor]):
        self.reset_lora(index)
        if self.tp_size > 1:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.output_dim
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            lora_b = (lora_b[0][:, start_idx:end_idx], lora_b[1][:, start_idx:end_idx])
        if lora_a[0] is not None:
            self.lora_a_stacked[0][index, 0, :lora_a[0].shape[1], :lora_a[0].shape[0]].copy_(lora_a[0].T, non_blocking=True)
            self.lora_b_stacked[0][index, 0, :lora_b[0].shape[1], :lora_b[0].shape[0]].copy_(lora_b[0].T, non_blocking=True)
        if lora_a[1] is not None:
            self.lora_a_stacked[1][index, 0, :lora_a[1].shape[1], :lora_a[1].shape[0]].copy_(lora_a[1].T, non_blocking=True)
            self.lora_b_stacked[1][index, 0, :lora_b[1].shape[1], :lora_b[1].shape[0]].copy_(lora_b[1].T, non_blocking=True)

    def apply_weights(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(self.base_layer.linear_weights, x, bias)
        _apply_lora_packed_nslice(x, self.lora_a_stacked, self.lora_b_stacked, self.indices[:self.indices_len[0]], output, (self.output_dim, self.output_dim))
        return output