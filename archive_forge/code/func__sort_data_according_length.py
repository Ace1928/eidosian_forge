import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _sort_data_according_length(input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Sort tokenized sentence from the shortest to the longest one."""
    sorted_indices = attention_mask.sum(1).argsort()
    input_ids = input_ids[sorted_indices]
    attention_mask = attention_mask[sorted_indices]
    return (input_ids, attention_mask, sorted_indices)