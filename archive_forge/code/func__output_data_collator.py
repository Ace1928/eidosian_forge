import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _output_data_collator(model_output: Tensor, attention_mask: Tensor, target_len: int) -> Tuple[Tensor, Tensor]:
    """Pad the model output and attention mask to the target length."""
    zeros_shape = list(model_output.shape)
    zeros_shape[2] = target_len - zeros_shape[2]
    model_output = torch.cat([model_output, torch.zeros(zeros_shape, dtype=model_output.dtype).to(model_output.device)], dim=2)
    zeros = torch.zeros(zeros_shape[0], zeros_shape[2], dtype=attention_mask.dtype).to(attention_mask.device)
    attention_mask = torch.cat([attention_mask, zeros], dim=1)
    return (model_output, attention_mask)