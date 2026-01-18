import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _shift_words(pred_words: List[str], target_words: List[str], cached_edit_distance: _LevenshteinEditDistance, checked_candidates: int) -> Tuple[int, List[str], int]:
    """Attempt to shift words to match a hypothesis with a reference.

    It returns the lowest number of required edits between a hypothesis and a provided reference, a list of shifted
    words and number of checked candidates. Note that the filtering of possible shifts and shift selection are heavily
    based on somewhat arbitrary heuristics. The code here follows as closely as possible the logic in Tercom, not
    always justifying the particular design choices.
    The paragraph copied from https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/lib_ter.py.

    Args:
        pred_words: A list of tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.
        cached_edit_distance: A pre-computed edit distance between a hypothesis and a reference.
        checked_candidates: A number of checked hypothesis candidates to match a provided reference.

    Return:
        best_score:
            The best (lowest) number of required edits to match hypothesis and reference sentences.
        shifted_words:
            A list of shifted words in hypothesis sentences.
        checked_candidates:
            A number of checked hypothesis candidates to match a provided reference.

    """
    edit_distance, inverted_trace = cached_edit_distance(pred_words)
    trace = _flip_trace(inverted_trace)
    alignments, target_errors, pred_errors = _trace_to_alignment(trace)
    best: Optional[Tuple[int, int, int, int, List[str]]] = None
    for pred_start, target_start, length in _find_shifted_pairs(pred_words, target_words):
        if _handle_corner_cases_during_shifting(alignments, pred_errors, target_errors, pred_start, target_start, length):
            continue
        prev_idx = -1
        for offset in range(-1, length):
            if target_start + offset == -1:
                idx = 0
            elif target_start + offset in alignments:
                idx = alignments[target_start + offset] + 1
            else:
                break
            if idx == prev_idx:
                continue
            prev_idx = idx
            shifted_words = _perform_shift(pred_words, pred_start, length, idx)
            candidate = (edit_distance - cached_edit_distance(shifted_words)[0], length, -pred_start, -idx, shifted_words)
            checked_candidates += 1
            if not best or candidate > best:
                best = candidate
        if checked_candidates >= _MAX_SHIFT_CANDIDATES:
            break
    if not best:
        return (0, pred_words, checked_candidates)
    best_score, _, _, _, shifted_words = best
    return (best_score, shifted_words, checked_candidates)