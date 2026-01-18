from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _calculate_fscore(matching_char_n_grams: Dict[int, Tensor], matching_word_n_grams: Dict[int, Tensor], hyp_char_n_grams: Dict[int, Tensor], hyp_word_n_grams: Dict[int, Tensor], ref_char_n_grams: Dict[int, Tensor], ref_word_n_grams: Dict[int, Tensor], n_order: float, beta: float) -> Tensor:
    """Calculate sentence-level chrF/chrF++ score.

    For given hypothesis and reference statistics (either sentence-level or corpus-level)
    the chrF/chrF++ score is returned.

    Args:
        matching_char_n_grams:
            A total number of matching character n-grams between the best matching reference and hypothesis.
        matching_word_n_grams:
            A total number of matching word n-grams between the best matching reference and hypothesis.
        hyp_char_n_grams: A total number of hypothesis character n-grams.
        hyp_word_n_grams: A total number of hypothesis word n-grams.
        ref_char_n_grams: A total number of reference character n-grams.
        ref_word_n_grams: A total number of reference word n-grams.
        n_order: A sum of character and word n-gram order.
        beta: A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.

    Return:
        A chrF/chrF++ score. This function is universal both for sentence-level and corpus-level calculation.

    """

    def _get_n_gram_fscore(matching_n_grams: Dict[int, Tensor], ref_n_grams: Dict[int, Tensor], hyp_n_grams: Dict[int, Tensor], beta: float) -> Dict[int, Tensor]:
        """Get n-gram level f-score."""
        precision: Dict[int, Tensor] = {n: matching_n_grams[n] / hyp_n_grams[n] if hyp_n_grams[n] > 0 else tensor(0.0) for n in matching_n_grams}
        recall: Dict[int, Tensor] = {n: matching_n_grams[n] / ref_n_grams[n] if ref_n_grams[n] > 0 else tensor(0.0) for n in matching_n_grams}
        denominator: Dict[int, Tensor] = {n: torch.max(beta ** 2 * precision[n] + recall[n], _EPS_SMOOTHING) for n in matching_n_grams}
        f_score: Dict[int, Tensor] = {n: (1 + beta ** 2) * precision[n] * recall[n] / denominator[n] for n in matching_n_grams}
        return f_score
    char_n_gram_f_score = _get_n_gram_fscore(matching_char_n_grams, ref_char_n_grams, hyp_char_n_grams, beta)
    word_n_gram_f_score = _get_n_gram_fscore(matching_word_n_grams, ref_word_n_grams, hyp_word_n_grams, beta)
    return (sum(char_n_gram_f_score.values()) + sum(word_n_gram_f_score.values())) / tensor(n_order)