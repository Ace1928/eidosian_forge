import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _process_attention_mask_for_special_tokens(attention_mask: Tensor) -> Tensor:
    """Process attention mask to be zero for special [CLS] and [SEP] tokens as they're not included in BERT score.

    Args:
        attention_mask: An attention mask to be returned, for example, by a `transformers` tokenizer.

    Return:
        A processed attention mask.

    """
    attention_mask[:, 0] = 0
    sep_token_position = _cumsum(attention_mask - 0.1, dim=-1).argmax(-1)
    attention_mask[torch.arange(attention_mask.size(0)).long(), sep_token_position] = 0
    return attention_mask