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
def _get_dataloader(input_ids: Tensor, attention_mask: Tensor, idf: bool, batch_size: int, num_workers: int) -> DataLoader:
    """Prepare dataloader.

    Args:
        input_ids:
            Indices of input sequence tokens in the vocabulary.
        attention_mask:
            Mask to avoid performing attention on padding token indices.
        idf:
            A bool indicating whether normalization using inverse document frequencies should be used.
        batch_size:
            A batch size used for model processing.
        num_workers:
            A number of workers to use for a dataloader.

    Return:
        An instance of ``torch.utils.data.DataLoader`` used for iterating over examples.

    """
    dataset = TokenizedDataset(input_ids, attention_mask, idf)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)