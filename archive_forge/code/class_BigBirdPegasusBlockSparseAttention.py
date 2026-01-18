import copy
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_bigbird_pegasus import BigBirdPegasusConfig
class BigBirdPegasusBlockSparseAttention(nn.Module):

    def __init__(self, config, seed=None):
        super().__init__()
        self.max_seqlen = config.max_position_embeddings
        self.seed = seed
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}.')
        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, band_mask=None, from_mask=None, to_mask=None, from_blocked_mask=None, to_blocked_mask=None, output_attentions=None):
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size
        if from_seq_length % from_block_size != 0:
            raise ValueError('Query sided sequence length must be multiple of block size')
        if to_seq_length % to_block_size != 0:
            raise ValueError('Key/Value sided sequence length must be multiple of block size')
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        context_layer, attention_probs = self.bigbird_block_sparse_attention(query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, self.num_attention_heads, self.num_random_blocks, self.attention_head_size, from_block_size, to_block_size, batch_size, from_seq_length, to_seq_length, seed=self.seed, plan_from_length=None, plan_num_rand_blocks=None, output_attentions=output_attentions)
        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication"""
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(inp_1.shape[:ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1]))

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication with transpose"""
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)).view(inp_1.shape[:ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    def bigbird_block_sparse_attention(self, query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, n_heads, n_rand_blocks, attention_head_size, from_block_size, to_block_size, batch_size, from_seq_len, to_seq_len, seed, plan_from_length, plan_num_rand_blocks, output_attentions):
        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        rsqrt_d = 1 / math.sqrt(attention_head_size)
        bsz = batch_size
        attn_mask_penalty = -10000.0
        np.random.seed(seed)
        if from_seq_len in [1024, 3072, 4096]:
            rand_attn = [self._bigbird_block_rand_mask(self.max_seqlen, self.max_seqlen, from_block_size, to_block_size, n_rand_blocks, last_idx=1024)[:from_seq_len // from_block_size - 2] for _ in range(n_heads)]
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(from_seq_len, from_block_size, n_rand_blocks)
            rand_attn = self._bigbird_block_rand_mask_with_head(from_seq_length=from_seq_len, to_seq_length=to_seq_len, from_block_size=from_block_size, to_block_size=to_block_size, num_heads=n_heads, plan_from_length=plan_from_length, plan_num_rand_blocks=plan_num_rand_blocks)
        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
        rand_attn.unsqueeze_(0)
        rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)
        rand_mask = self._create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size)
        blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
        gathered_key = gathered_key.view(bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)
        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = nn.functional.softmax(first_product, dim=-1)
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)
        second_key_mat = torch.cat([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1], blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1], gathered_key[:, :, 0]], dim=2)
        second_value_mat = torch.cat([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1], blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1], gathered_value[:, :, 0]], dim=2)
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat([to_mask[:, :, :, :3 * to_block_size], to_mask[:, :, :, -to_block_size:], to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size])], dim=3)
        second_rand_pad = torch.cat([rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]), rand_mask[:, :, 0]], dim=3)
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = nn.functional.softmax(second_product, dim=-1)
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)
        second_context_layer.unsqueeze_(2)
        exp_blocked_key_matrix = torch.cat([blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3)
        exp_blocked_value_matrix = torch.cat([blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]], dim=3)
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        inner_band_product = inner_band_product * rsqrt_d
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
        rand_band_product = rand_band_product * rsqrt_d
        first_band_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, 0])
        first_band_product = first_band_product * rsqrt_d
        last_band_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, -1])
        last_band_product = last_band_product * rsqrt_d
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty
        band_product = torch.cat([first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1)
        attn_weights = nn.functional.softmax(band_product, dim=-1)
        context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, to_block_size:4 * to_block_size], exp_blocked_value_matrix, ndim=5)
        context_layer += self.torch_bmm_nd(attn_weights[:, :, :, :, 4 * to_block_size:-to_block_size], gathered_value[:, :, 1:-1], ndim=5)
        context_layer += torch.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0])
        context_layer += torch.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1])
        second_last_key_mat = torch.cat([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3], blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1], gathered_key[:, :, -1]], dim=2)
        second_last_value_mat = torch.cat([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3], blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1], gathered_value[:, :, -1]], dim=2)
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat([to_mask[:, :, :, :to_block_size], to_mask[:, :, :, -3 * to_block_size:], to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size])], dim=3)
        second_last_rand_pad = torch.cat([rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]), rand_mask[:, :, -1]], dim=3)
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = nn.functional.softmax(second_last_product, dim=-1)
        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer.unsqueeze_(2)
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = nn.functional.softmax(last_product, dim=-1)
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)
        context_layer = torch.cat([first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer], dim=2)
        context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)
        if output_attentions:
            attention_probs = torch.zeros(bsz, n_heads, from_seq_len, to_seq_len, dtype=torch.float, device=context_layer.device)
            attention_probs[:, :, :from_block_size, :] = first_attn_weights
            attention_probs[:, :, from_block_size:2 * from_block_size, :3 * to_block_size] = second_attn_weights[:, :, :, :3 * to_block_size]
            attention_probs[:, :, from_block_size:2 * from_block_size, -to_block_size:] = second_attn_weights[:, :, :, 3 * to_block_size:4 * to_block_size]
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_attn_weights):
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                    right_slice = w2[:, 4 * to_block_size:]
                    attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
            for q_idx in range(from_seq_len // from_block_size - 4):
                attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)[:, :, 2:-2, :, 1:-1, :]
                right_slice = attn_weights[:, :, q_idx, :, to_block_size:4 * to_block_size]
                attn_probs_view[:, :, q_idx, :, q_idx:q_idx + 3, :] = right_slice.view(bsz, n_heads, from_block_size, 3, to_block_size)
            attention_probs[:, :, 2 * from_block_size:-2 * from_block_size, :to_block_size] = attn_weights[:, :, :, :, :to_block_size].view(bsz, n_heads, -1, to_block_size)
            attention_probs[:, :, 2 * from_block_size:-2 * from_block_size, -to_block_size:] = attn_weights[:, :, :, :, -to_block_size:].view(bsz, n_heads, -1, to_block_size)
            for p1, i1, w1 in zip(range(bsz), rand_attn, attn_weights):
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    for q_idx in range(1, len(i2) - 1):
                        attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                        right_slice = w2[q_idx - 1, :, 4 * to_block_size:-to_block_size]
                        attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
            attention_probs[:, :, -2 * from_block_size:-from_block_size, :to_block_size] = second_last_attn_weights[:, :, :, :to_block_size]
            attention_probs[:, :, -2 * from_block_size:-from_block_size, -3 * to_block_size:] = second_last_attn_weights[:, :, :, to_block_size:4 * to_block_size]
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_last_attn_weights):
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    attn_probs_view = attention_probs.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, to_seq_len // to_block_size, to_block_size)
                    right_slice = w2[:, 4 * to_block_size:]
                    attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(from_block_size, n_rand_blocks, to_block_size)
            attention_probs[:, :, -from_block_size:, :] = last_attn_weights
        else:
            attention_probs = None
        return (context_layer, attention_probs)

    @staticmethod
    def torch_gather_b2(params, indices):
        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(f'Make sure that the first two dimensions of params and indices are identical,                 but they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}')
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        num_indices_to_pick_from = params.shape[2]
        shift = torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
        indices_shift = torch.div(shift, num_indices_to_gather, rounding_mode='floor') * num_indices_to_pick_from
        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])
        out_flattened = flattened_params.index_select(0, flattened_indices)
        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, num_attention_heads, num_rand_blocks, batch_size, from_seq_length, from_block_size):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
            rand_attn: [batch_size, num_attention_heads,
            from_seq_length//from_block_size-2, num_rand_blocks]
            num_attention_heads: int. Number of attention heads.
            num_rand_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
        """
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = torch.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        rand_mask = torch.einsum('blq,bhlk->bhlqk', from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        Gives the plan of where to put random attention.

        Args:
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.
            num_rand_blocks: int. Number of random chunks per row.

        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
            each block
        """
        plan_from_length = []
        plan_num_rand_blocks = []
        if 2 * num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - num_rand_blocks // 2)
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)
        return (plan_from_length, plan_num_rand_blocks)

    def _bigbird_block_rand_mask(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        if not self.training:
            return rand_attn
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > 2 * to_block_size:
            last = last_idx // to_block_size - 1
        r = num_rand_blocks
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif end + 1 == last:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(np.concatenate((middle_seq[:start], middle_seq[end + 1:last])))[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_heads, plan_from_length, plan_num_rand_blocks, window_block_left=1, window_block_right=1, global_block_top=1, global_block_bottom=1, global_block_left=1, global_block_right=1):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_heads: int. total number of heads.
            plan_from_length: list. plan from length where num_random_blocks are chosen from.
            plan_num_rand_blocks: list. number of rand blocks within the plan.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_top: int. number of blocks at the top.
            global_block_bottom: int. number of blocks at the bottom.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            adjacency list of size num_head where each element is of size from_seq_length//from_block_size-2 by
            num_rand_blocks
        """
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        if from_seq_length not in plan_from_length:
            raise ValueError('Error from sequence length not in plan!')
        num_blocks = from_seq_length // from_block_size
        plan_block_length = np.array(plan_from_length) // from_block_size
        max_plan_idx = plan_from_length.index(from_seq_length)
        rand_attn = [np.zeros((num_blocks, np.sum(plan_num_rand_blocks[:max_plan_idx + 1])), dtype=np.int32) for i in range(num_heads)]
        if not self.training:
            for nh in range(num_heads):
                rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
            return rand_attn
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=plan_block_length[plan_idx - 1], to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right)
                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[pl_id], num_rand_blocks=plan_num_rand_blocks[pl_id], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right)
            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]
            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right)
        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(block_id, to_start_block_id, to_end_block_id, num_rand_blocks, window_block_left=1, window_block_right=1, global_block_left=1, global_block_right=1):
        """
        For a single row block get random row attention.

        Args:
            block_id: int. block id of row.
            to_start_block_id: int. random attention column start id.
            to_end_block_id: int. random attention column end id.
            num_rand_blocks: int. number of random blocks to be selected.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            row containing the random attention vector of size num_rand_blocks.
        """
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        perm_block = np.random.permutation(to_block_list)
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)
        selected_random_blokcs = []
        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)