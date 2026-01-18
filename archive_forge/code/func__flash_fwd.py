import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _flash_fwd(query, key, value, cu_seq_lens_q, cu_seq_lens_k, seqused_k, max_seq_len_q, max_seq_len_k, p, softmax_scale, is_causal, window_left, window_right, return_softmax):
    if cu_seq_lens_q is None:
        assert cu_seq_lens_k is None
        assert seqused_k is None
        out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state = _C_flashattention.fwd(query, key, value, None, None, p, softmax_scale, is_causal, window_left, window_right, return_softmax, None)
    else:
        out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state = _C_flashattention.varlen_fwd(query, key, value, None, cu_seq_lens_q, cu_seq_lens_k, seqused_k, None, max_seq_len_q, max_seq_len_k, p, softmax_scale, False, is_causal, window_left, window_right, return_softmax, None)
    return (out, softmax_lse, rng_state)