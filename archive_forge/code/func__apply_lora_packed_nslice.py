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
def _apply_lora_packed_nslice(x: torch.Tensor, lora_a_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], lora_b_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], indices: torch.Tensor, output: torch.Tensor, output_slices: Tuple[int, ...]):
    """Applies lora to each input.

    This method applies all loras to each input. It uses the
    indices vector to determine which lora yields the
    correct output. An index of -1 means no lora should be
    applied. This method adds the final lora results to the
    output.

    This method is used for layers that are composed of multiple sublayers
    (slices) packed together.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        lora_a_stacked:    3 element tuple of (num_loras, lora_rank, hidden_dim)
        lora_b_stacked:    3 element tuple of (num_loras, output_dim, lora_rank)
        indices:           (batch_size)
        output:            (batch_size, q_slice_size + 2*kv_slice_size)
        output_slices:     n-1 element tuple of (slice_size...), where n is number of slices
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    offset_left = 0
    for slice_idx in range(len(output_slices)):
        add_lora_slice(output, x, lora_a_stacked[slice_idx], lora_b_stacked[slice_idx], indices, 0, 1.0, offset_left, output_slices[slice_idx])
        offset_left += output_slices[slice_idx]
    return output.view_as(org_output)