import math
import hydra
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn.flash_blocksparse_attn_interface import (
Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many query each sequence in the batch consists of
        