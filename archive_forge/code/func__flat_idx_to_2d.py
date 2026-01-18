from functools import reduce
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from .base_sparsifier import BaseSparsifier
def _flat_idx_to_2d(idx, shape):
    rows = idx // shape[1]
    cols = idx % shape[1]
    return (rows, cols)