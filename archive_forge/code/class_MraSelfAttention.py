import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mra import MraConfig
class MraSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        kernel_loaded = mra_cuda_kernel is not None
        if is_torch_cuda_available() and is_ninja_available() and (not kernel_loaded):
            try:
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f'Could not load the custom kernel for multi-scale deformable attention: {e}')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        self.num_block = config.max_position_embeddings // 32 * config.block_per_row
        self.num_block = min(self.num_block, int((config.max_position_embeddings // 32) ** 2))
        self.approx_mode = config.approx_mode
        self.initial_prior_first_n_blocks = config.initial_prior_first_n_blocks
        self.initial_prior_diagonal_n_blocks = config.initial_prior_diagonal_n_blocks

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        batch_size, num_heads, seq_len, head_dim = query_layer.size()
        attention_mask = 1.0 + attention_mask / 10000.0
        attention_mask = attention_mask.squeeze().repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len).int()
        gpu_warp_size = 32
        if head_dim < gpu_warp_size:
            pad_size = (batch_size, num_heads, seq_len, gpu_warp_size - head_dim)
            query_layer = torch.cat([query_layer, torch.zeros(pad_size, device=query_layer.device)], dim=-1)
            key_layer = torch.cat([key_layer, torch.zeros(pad_size, device=key_layer.device)], dim=-1)
            value_layer = torch.cat([value_layer, torch.zeros(pad_size, device=value_layer.device)], dim=-1)
        context_layer = mra2_attention(query_layer.float(), key_layer.float(), value_layer.float(), attention_mask.float(), self.num_block, approx_mode=self.approx_mode, initial_prior_first_n_blocks=self.initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks=self.initial_prior_diagonal_n_blocks)
        if head_dim < gpu_warp_size:
            context_layer = context_layer[:, :, :, :head_dim]
        context_layer = context_layer.reshape(batch_size, num_heads, seq_len, head_dim)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer,)
        return outputs