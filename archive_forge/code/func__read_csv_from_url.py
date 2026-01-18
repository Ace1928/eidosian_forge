import csv
import urllib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _read_csv_from_url(baseline_url: str) -> Tensor:
    """Read baseline from csv file from URL.

    This method is implemented to avoid `pandas` dependency.

    """
    with urllib.request.urlopen(baseline_url) as http_request:
        baseline_list = [[float(item) for item in row.strip().decode('utf-8').split(',')] for idx, row in enumerate(http_request) if idx > 0]
        return torch.tensor(baseline_list)[:, 1:]