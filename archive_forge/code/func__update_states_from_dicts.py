import itertools
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics import Metric
from torchmetrics.functional.text.chrf import _chrf_score_compute, _chrf_score_update, _prepare_n_grams_dicts
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _update_states_from_dicts(self, n_grams_dicts_tuple: _DICT_STATES_TYPES) -> None:
    """Update global metric states based on the n-gram dictionaries calculated on the current batch."""
    n_grams_dicts = dict(zip(_DICT_STATES_NAMES, n_grams_dicts_tuple))
    for (n_gram_level, n_gram_order), text in self._get_text_n_gram_iterator():
        for n in range(1, n_gram_order + 1):
            dict_name = self._get_dict_name(text, n_gram_level)
            state_name = self._get_state_name(text, n_gram_level, n)
            setattr(self, state_name, n_grams_dicts[dict_name][n])