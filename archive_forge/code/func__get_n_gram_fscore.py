from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _get_n_gram_fscore(matching_n_grams: Dict[int, Tensor], ref_n_grams: Dict[int, Tensor], hyp_n_grams: Dict[int, Tensor], beta: float) -> Dict[int, Tensor]:
    """Get n-gram level f-score."""
    precision: Dict[int, Tensor] = {n: matching_n_grams[n] / hyp_n_grams[n] if hyp_n_grams[n] > 0 else tensor(0.0) for n in matching_n_grams}
    recall: Dict[int, Tensor] = {n: matching_n_grams[n] / ref_n_grams[n] if ref_n_grams[n] > 0 else tensor(0.0) for n in matching_n_grams}
    denominator: Dict[int, Tensor] = {n: torch.max(beta ** 2 * precision[n] + recall[n], _EPS_SMOOTHING) for n in matching_n_grams}
    f_score: Dict[int, Tensor] = {n: (1 + beta ** 2) * precision[n] * recall[n] / denominator[n] for n in matching_n_grams}
    return f_score