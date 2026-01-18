import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _compute_sentence_statistics(pred_words: List[str], target_words: List[List[str]]) -> Tuple[Tensor, Tensor]:
    """Compute sentence TER statistics between hypothesis and provided references.

    Args:
        pred_words: A list of tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.

    Return:
        best_num_edits:
            The best (lowest) number of required edits to match hypothesis and reference sentences.
        avg_tgt_len:
            Average length of tokenized reference sentences.

    """
    tgt_lengths = tensor(0.0)
    best_num_edits = tensor(2e+16)
    for tgt_words in target_words:
        num_edits = _translation_edit_rate(tgt_words, pred_words)
        tgt_lengths += len(tgt_words)
        if num_edits < best_num_edits:
            best_num_edits = num_edits
    avg_tgt_len = tgt_lengths / len(target_words)
    return (best_num_edits, avg_tgt_len)