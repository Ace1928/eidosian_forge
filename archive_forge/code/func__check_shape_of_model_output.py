import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _check_shape_of_model_output(output: Tensor, input_ids: Tensor) -> None:
    """Check if the shape of the user's own model output."""
    bs, seq_len = input_ids.shape[:2]
    invalid_out_shape = len(output.shape) != 3 or output.shape[0] != bs or output.shape[1] != seq_len
    if invalid_out_shape:
        raise ValueError(f'The model output must be `Tensor` of a shape `[batch_size, seq_len, model_dim]` i.e. [{bs}, {seq_len}. , `model_dim`], but got {output.shape}.')