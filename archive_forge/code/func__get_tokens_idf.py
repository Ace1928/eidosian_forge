import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _get_tokens_idf(self) -> Dict[int, float]:
    """Calculate token inverse document frequencies.

        Return:
            A python dictionary containing inverse document frequencies for token ids.

        """
    token_counter: Counter = Counter()
    for tokens in map(self._set_of_tokens, self.text['input_ids']):
        token_counter.update(tokens)
    tokens_idf: Dict[int, float] = defaultdict(self._get_tokens_idf_default_value)
    tokens_idf.update({idx: math.log((self.num_sentences + 1) / (occurrence + 1)) for idx, occurrence in token_counter.items()})
    return tokens_idf