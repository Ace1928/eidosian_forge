from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _prepare_n_grams_dicts(n_char_order: int, n_word_order: int) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]]:
    """Prepare dictionaries with default zero values for total ref, hypothesis and matching character and word n-grams.

    Args:
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.

    Return:
        Dictionaries with default zero values for total reference, hypothesis and matching character and word
        n-grams.

    """
    total_preds_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_preds_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}
    total_target_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_target_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}
    total_matching_char_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_char_order)}
    total_matching_word_n_grams: Dict[int, Tensor] = {n + 1: tensor(0.0) for n in range(n_word_order)}
    return (total_preds_char_n_grams, total_preds_word_n_grams, total_target_char_n_grams, total_target_word_n_grams, total_matching_char_n_grams, total_matching_word_n_grams)