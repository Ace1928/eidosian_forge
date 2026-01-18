import os
from enum import unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from typing_extensions import Literal
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
def _infolm_update(preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]], tokenizer: 'PreTrainedTokenizerBase', max_length: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Update the metric state by a tokenization of ``preds`` and ``target`` sentencens.

    Args:
        preds:
            An iterable of hypothesis corpus.
        target:
            An iterable of reference corpus.
        tokenizer:
            Initialized tokenizer from HuggingFace's `transformers package.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.

    Return:
        Tokenizerd ``preds`` and ``target`` sentences represented with ``input_ids`` and ``attention_mask`` tensors.

    """
    if not isinstance(preds, (str, list)):
        preds = list(preds)
    if not isinstance(target, (str, list)):
        target = list(target)
    preds_input = tokenizer(preds, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
    target_input = tokenizer(target, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
    return (preds_input.input_ids, preds_input.attention_mask, target_input.input_ids, target_input.attention_mask)