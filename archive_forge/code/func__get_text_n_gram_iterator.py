import itertools
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics import Metric
from torchmetrics.functional.text.chrf import _chrf_score_compute, _chrf_score_update, _prepare_n_grams_dicts
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _get_text_n_gram_iterator(self) -> Iterator[Tuple[Tuple[str, int], str]]:
    """Get iterator over char/word and reference/hypothesis/matching n-gram level."""
    return itertools.product(zip(_N_GRAM_LEVELS, [self.n_char_order, self.n_word_order]), _TEXT_LEVELS)