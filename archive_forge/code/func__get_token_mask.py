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
def _get_token_mask(input_ids: Tensor, pad_token_id: int, sep_token_id: int, cls_token_id: int) -> Tensor:
    """Generate a token mask for differentiating all special tokens in the input batch.

    There are 0s for special tokens and 1s otherwise.

    Args:
        input_ids:
            Indices of input sequence tokens in the vocabulary.
        pad_token_id:
            An id of ``<PAD>`` tokens that are used to make arrays of tokens the same size for batching purpose
        cls_token_id:
            An id of ``<CLS>`` token that represents the class of the input. (It might be ``<BOS>`` token for some
            models.)
        sep_token_id:
            An id of ``<SEP>`` token that separates two different sentences in the same input. (It might be ``<EOS>``
            token for some models.)

    Return:
        Tensor mask of 0s and 1s that masks all special tokens in the ``input_ids`` tensor.

    """
    token_mask = input_ids.eq(pad_token_id) | input_ids.eq(sep_token_id) | input_ids.eq(cls_token_id)
    return ~token_mask