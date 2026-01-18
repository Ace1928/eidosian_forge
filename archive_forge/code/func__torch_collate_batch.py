import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int]=None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all((x.size(0) == length_of_first for x in examples))
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)
    if tokenizer._pad_token is None:
        raise ValueError(f'You are attempting to pad samples but the tokenizer you are using ({tokenizer.__class__.__name__}) does not have a pad token.')
    max_length = max((x.size(0) for x in examples))
    if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
        max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == 'right':
            result[i, :example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result