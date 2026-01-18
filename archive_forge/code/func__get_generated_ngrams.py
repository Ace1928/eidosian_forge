import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])