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
def _get_char_input_ids(self, input_ids, subwords_batch, char_count_per_id, pad_token_id=0, unk_token_id=1):
    """
        Returns the corresponding character input id for each character of `subwords_batch`.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            char_count_per_id (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Number of characters per input id.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
        Returns:
            `torch.Tensor`: Tensor of shape `(batch_size, char_sequence_length)` containing the id of each character.
        """
    if not hasattr(self.generation_config, 'char_to_id'):
        raise ValueError("This model generation config doesn't have a `char_to_id` key which maps\n                characters to character ids. Make sure to load the right generation config.")
    batch_size = input_ids.shape[0]
    max_len = int(char_count_per_id.sum(1).max().item())
    char_seqs = input_ids.new_zeros((batch_size, max_len)).fill_(pad_token_id)
    subword_lens = input_ids.ne(pad_token_id).sum(1)
    for batch_id in range(batch_size):
        total = 0
        subword_indices = input_ids[batch_id, :subword_lens[batch_id]]
        subwords = subwords_batch[batch_id][:subword_lens[batch_id]]
        for subword_idx, subword in zip(subword_indices, subwords):
            if subword_idx == unk_token_id:
                char_ids = [unk_token_id]
            else:
                char_ids = [self.generation_config.char_to_id.get(ch, unk_token_id) for ch in list(subword)]
            char_seq_len = len(char_ids)
            char_seqs[batch_id, total:total + char_seq_len] = torch.tensor(char_ids).to(char_seqs)
            total += char_seq_len
    return char_seqs