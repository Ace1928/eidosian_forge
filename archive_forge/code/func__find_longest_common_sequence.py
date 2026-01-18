import json
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple, Union
import numpy as np
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
def _find_longest_common_sequence(sequences, token_timestamp_sequences=None):
    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []
    if token_timestamp_sequences:
        left_token_timestamp_sequence = token_timestamp_sequences[0]
        total_token_timestamp_sequence = []
    for seq_idx, right_sequence in enumerate(sequences[1:]):
        max_ = 0.0
        max_indices = (left_length, left_length, 0, 0)
        right_length = len(right_sequence)
        for i in range(1, left_length + right_length):
            eps = i / 10000.0
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])
            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])
            if len(left) != len(right):
                raise RuntimeError('There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference.')
            matches = np.sum(left == right)
            matching = matches / i + eps
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)
        left_start, left_stop, right_start, right_stop = max_indices
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)
        if token_timestamp_sequences:
            total_token_timestamp_sequence.extend(left_token_timestamp_sequence[:left_mid])
            left_token_timestamp_sequence = token_timestamp_sequences[seq_idx + 1][right_mid:]
    total_sequence.extend(left_sequence)
    if token_timestamp_sequences is None:
        return total_sequence
    if len(token_timestamp_sequences) > 0:
        total_token_timestamp_sequence.extend(left_token_timestamp_sequence)
        return (total_sequence, total_token_timestamp_sequence)
    else:
        return (total_sequence, [])