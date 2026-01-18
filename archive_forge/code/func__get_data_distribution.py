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
@torch.no_grad()
def _get_data_distribution(model: 'PreTrainedModel', dataloader: DataLoader, temperature: float, idf: bool, special_tokens_map: Dict[str, int], verbose: bool) -> Tensor:
    """Calculate a discrete probability distribution according to the methodology described in `InfoLM`_.

    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        dataloader:
            An instance of `torch.utils.data.DataLoader` used for iterating over examples.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        special_tokens_map:
            A dictionary mapping tokenizer special tokens into the corresponding integer values.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.

    Return:
        A discrete probability distribution.

    """
    device = model.device
    prob_distribution: List[Tensor] = []
    for batch in _get_progress_bar(dataloader, verbose):
        batch = _input_data_collator(batch, device)
        prob_distribution.append(_get_batch_distribution(model, batch, temperature, idf, special_tokens_map))
    return torch.cat(prob_distribution, dim=0)