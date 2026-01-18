import itertools
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics import Metric
from torchmetrics.functional.text.chrf import _chrf_score_compute, _chrf_score_update, _prepare_n_grams_dicts
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
@staticmethod
def _get_state_name(text: str, n_gram_level: str, n: int) -> str:
    """Return a metric state name w.r.t input args."""
    return f'total_{text}_{n_gram_level}_{n}_grams'