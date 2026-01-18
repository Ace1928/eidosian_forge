from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _calculate_sentence_level_chrf_score(targets: List[str], pred_char_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]], pred_word_n_grams_counts: Dict[int, Dict[Tuple[str, ...], Tensor]], pred_char_n_grams: Dict[int, Tensor], pred_word_n_grams: Dict[int, Tensor], n_char_order: int, n_word_order: int, n_order: float, beta: float, lowercase: bool, whitespace: bool) -> Tuple[Tensor, Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]]:
    """Calculate the best sentence-level chrF/chrF++ score.

    For a given pre-processed hypothesis, all references are evaluated and score and statistics
    for the best matching reference is returned.

    Args:
        targets: An iterable of references.
        pred_char_n_grams_counts: A dictionary of dictionaries with hypothesis character n-grams.
        pred_word_n_grams_counts: A dictionary of dictionaries with hypothesis word n-grams.
        pred_char_n_grams: A total number of hypothesis character n-grams.
        pred_word_n_grams: A total number of hypothesis word n-grams.
        n_char_order: A character n-gram order.
        n_word_order: A word n-gram order.
        n_order: A sum of character and word n-gram order.
        beta: A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.

    Return:
        Return chrF/chrF++ score and statistics for the best matching hypothesis and reference.

        f_score: A sentence-level chrF/chrF++ score.
        matching_char_n_grams:
            A total number of matching character n-grams between the best matching reference and hypothesis.
        matching_word_n_grams:
            A total number of matching word n-grams between the best matching reference and hypothesis.
        target_char_n_grams: A total number of reference character n-grams.
        target_word_n_grams: A total number of reference word n-grams.

    """
    best_f_score = tensor(0.0)
    best_matching_char_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    best_matching_word_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    best_target_char_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    best_target_word_n_grams: Dict[int, Tensor] = defaultdict(lambda: tensor(0.0))
    for target in targets:
        target_char_n_grams_counts, target_word_n_grams_counts, target_char_n_grams, target_word_n_grams = _get_n_grams_counts_and_total_ngrams(target, n_char_order, n_word_order, lowercase, whitespace)
        matching_char_n_grams = _get_ngram_matches(target_char_n_grams_counts, pred_char_n_grams_counts)
        matching_word_n_grams = _get_ngram_matches(target_word_n_grams_counts, pred_word_n_grams_counts)
        f_score = _calculate_fscore(matching_char_n_grams, matching_word_n_grams, pred_char_n_grams, pred_word_n_grams, target_char_n_grams, target_word_n_grams, n_order, beta)
        if f_score > best_f_score:
            best_f_score = f_score
            best_matching_char_n_grams = matching_char_n_grams
            best_matching_word_n_grams = matching_word_n_grams
            best_target_char_n_grams = target_char_n_grams
            best_target_word_n_grams = target_word_n_grams
    return (best_f_score, best_matching_char_n_grams, best_matching_word_n_grams, best_target_char_n_grams, best_target_word_n_grams)