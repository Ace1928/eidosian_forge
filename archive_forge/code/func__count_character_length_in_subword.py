import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
def _count_character_length_in_subword(self, input_ids, subwords_batch, merge_space_with_prev_subword=False, pad_token_id=0, unk_token_id=1, space='▁'):
    """
        Counts the number of characters per text string associated with the input token id.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            merge_space_with_prev_subword (`bool`, *optional*, defaults to `False`):
                Indicates if the space character is merged with the previous subword. If `False`, it will be merged
                with the next subword.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
            space (`str`, *optional*, defaults to `"▁"`):
                The space character.
        """
    batch_size, _ = input_ids.shape
    char_count_per_id = input_ids.new_zeros(input_ids.size())
    subword_lens = input_ids.ne(pad_token_id).sum(1)
    for batch_id in range(batch_size):
        subword_indices = input_ids[batch_id, :subword_lens[batch_id]]
        subwords = subwords_batch[batch_id][:subword_lens[batch_id]]
        is_next_start_with_space = [len(subwords[i + 1]) > 1 and subwords[i + 1][0] == space if i < len(subwords) - 1 else False for i in range(len(subwords))]
        is_punc = [len(subwords[i]) == 1 and (not subwords[i].isalpha()) and (not subwords[i].isnumeric()) and (subwords[i] != space) for i in range(len(subwords))]
        for i, (subword_idx, subword) in enumerate(zip(subword_indices, subwords)):
            if subword_idx == pad_token_id:
                break
            if subword_idx == unk_token_id:
                char_len = 1
                if merge_space_with_prev_subword and is_next_start_with_space[i]:
                    char_len += 1
            else:
                char_len = len(subword)
                if merge_space_with_prev_subword:
                    if is_next_start_with_space[i]:
                        char_len += 1
                    if i > 0 and is_next_start_with_space[i - 1]:
                        char_len -= 1
                elif is_punc[i] and is_next_start_with_space[i]:
                    char_len += 1
                elif i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]:
                    char_len -= 1
            char_count_per_id[batch_id, i] = char_len
    return char_count_per_id