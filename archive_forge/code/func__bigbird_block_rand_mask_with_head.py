import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
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