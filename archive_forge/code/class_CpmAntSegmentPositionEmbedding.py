import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
class CpmAntSegmentPositionEmbedding(nn.Module):

    def __init__(self, config: CpmAntConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_buckets = config.position_bias_num_buckets
        self.max_distance = config.position_bias_max_distance
        self.num_segments = config.segment_types
        self.relative_attention_bias = nn.Parameter(torch.empty(config.segment_types * config.segment_types + config.position_bias_num_buckets, config.num_attention_heads))

    def forward(self, key_pos: torch.Tensor, query_pos: torch.Tensor, key_segment: torch.Tensor, query_segment: torch.Tensor):
        with torch.no_grad():
            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)
            if key_pos.size(0) != query_pos.size(0):
                raise AssertionError(f'key_pos.size(0) should be equal to query_pos.size(0), but got {key_pos.size(0)} and {query_pos.size(0)}!')
            if keylen != key_segment.size(1) or querylen != query_segment.size(1):
                raise AssertionError(f'keylen should be equal to key_segment.size(1), but got {keylen} and {key_segment.size(1)}!')
            if querylen != query_segment.size(1):
                raise AssertionError(f'querylen should be equal to query_segment.size(1), but got {querylen} and {query_segment.szie(1)}!')
            key_pos = key_pos.view(batch, -1, keylen)
            query_pos = query_pos.view(batch, querylen, -1)
            key_segment = key_segment.view(batch, -1, keylen)
            query_segment = query_segment.view(batch, querylen, -1)
            relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
            relative_position_bucket = relative_position_bucket + self.num_buckets
            absolute_position_bucket = self._position_bucket(torch.arange(keylen, dtype=torch.int32, device=relative_position_bucket.device)[None, :] - torch.arange(querylen, dtype=torch.int32, device=relative_position_bucket.device)[:, None], num_buckets=self.num_buckets, max_distance=self.max_distance)
            relative_position_bucket = torch.where(key_segment == query_segment, absolute_position_bucket[None, :, :], relative_position_bucket)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.int32)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        return relative_buckets