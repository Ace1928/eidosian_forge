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
def _apply_lora(x: torch.Tensor, lora_a_stacked: torch.Tensor, lora_b_stacked: torch.Tensor, indices: torch.Tensor, output: torch.Tensor):
    """Applies lora to each input.

    This method applies all loras to each input. It uses the
    indices vector to determine which lora yields the
    correct output. An index of -1 means no lora should be
    applied. This method adds the final lora results to the
    output.

    Input shapes:
        x:               (batch_size, hidden_dim)
        lora_a_stacked:  (num_loras, lora_rank, hidden_dim)
        lora_b_stacked:  (num_loras, output_dim, lora_rank)
        indices:         (batch_size)
        output:          (batch_size, output_dim)
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    add_lora(output, x, lora_a_stacked, lora_b_stacked, indices, 0, 1.0)
    return output.view_as(org_output)