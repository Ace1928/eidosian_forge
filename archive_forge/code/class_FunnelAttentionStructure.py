import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """
    cls_token_type_id: int = 2

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        self.pooling_mult = None

    def init_attention_inputs(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor]:
        """Returns the attention inputs associated to the inputs of the model."""
        self.pooling_mult = 1
        self.seq_len = seq_len = inputs_embeds.size(1)
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        cls_mask = nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0)) if self.config.separate_cls else None
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """Convert `token_type_ids` to `token_type_mat`."""
        token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        cls_ids = token_type_ids == self.cls_token_type_id
        cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        return cls_mat | token_type_mat

    def get_position_embeds(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> Union[Tuple[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shift attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        d_model = self.config.d_model
        if self.config.attention_type == 'factorized':
            pos_seq = torch.arange(0, seq_len, 1.0, dtype=torch.int64, device=device).to(dtype)
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=torch.int64, device=device).to(dtype)
            inv_freq = 1 / 10000 ** (freq_seq / (d_model // 2))
            sinusoid = pos_seq[:, None] * inv_freq[None]
            sin_embed = torch.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed)
            cos_embed = torch.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed)
            phi = torch.cat([sin_embed_d, sin_embed_d], dim=-1)
            psi = torch.cat([cos_embed, sin_embed], dim=-1)
            pi = torch.cat([cos_embed_d, cos_embed_d], dim=-1)
            omega = torch.cat([-sin_embed, cos_embed], dim=-1)
            return (phi, pi, psi, omega)
        else:
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=torch.int64, device=device).to(dtype)
            inv_freq = 1 / 10000 ** (freq_seq / (d_model // 2))
            rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=torch.int64, device=device).to(dtype)
            zero_offset = seq_len * 2
            sinusoid = rel_pos_id[:, None] * inv_freq[None]
            sin_embed = self.sin_dropout(torch.sin(sinusoid))
            cos_embed = self.cos_dropout(torch.cos(sinusoid))
            pos_embed = torch.cat([sin_embed, cos_embed], dim=-1)
            pos = torch.arange(0, seq_len, dtype=torch.int64, device=device).to(dtype)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.config.num_blocks):
                if block_index == 0:
                    position_embeds_pooling = None
                else:
                    pooled_pos = self.stride_pool_pos(pos, block_index)
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                    rel_pos = rel_pos[:, None] + zero_offset
                    rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                    position_embeds_pooling = torch.gather(pos_embed, 0, rel_pos)
                pos = pooled_pos
                stride = 2 ** block_index
                rel_pos = self.relative_pos(pos, stride)
                rel_pos = rel_pos[:, None] + zero_offset
                rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                position_embeds_no_pooling = torch.gather(pos_embed, 0, rel_pos)
                position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id: torch.Tensor, block_index: int):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        if self.config.separate_cls:
            cls_pos = pos_id.new_tensor([-2 ** block_index + 1])
            pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos: torch.Tensor, stride: int, pooled_pos=None, shift: int=1) -> torch.Tensor:
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos
        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]
        return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long, device=pos.device)

    def stride_pool(self, tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]], axis: Union[int, Tuple[int], List[int]]) -> torch.Tensor:
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor
        if isinstance(tensor, (tuple, list)):
            return type(tensor)((self.stride_pool(x, axis) for x in tensor))
        axis %= tensor.ndim
        axis_slice = slice(None, -1, 2) if self.config.separate_cls and self.config.truncate_seq else slice(None, None, 2)
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.config.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = torch.cat([tensor[cls_slice], tensor], axis=axis)
        return tensor[enc_slice]

    def pool_tensor(self, tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]], mode: str='mean', stride: int=2) -> torch.Tensor:
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None
        if isinstance(tensor, (tuple, list)):
            return type(tensor)((self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor))
        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = torch.cat([tensor[:, :1], suffix], dim=1)
        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]
        stride = (stride, 1)
        if mode == 'mean':
            tensor = nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == 'max':
            tensor = nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == 'min':
            tensor = -nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")
        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_pooling(self, output, attention_inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            if self.config.attention_type == 'factorized':
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        else:
            self.pooling_mult *= 2
            if self.config.attention_type == 'factorized':
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode='min')
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return (output, attention_inputs)

    def post_attention_pooling(self, attention_inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            self.pooling_mult *= 2
            if self.config.attention_type == 'factorized':
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode='min')
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs