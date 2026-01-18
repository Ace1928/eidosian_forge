from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _ngram_counts(char_or_word_list: List[str], n_gram_order: int) -> Dict[int, Dict[Tuple[str, ...], Tensor]]:
    """Calculate n-gram counts.

    Args:
        char_or_word_list: A list of characters of words
        n_gram_order: The largest number of n-gram.

    Return:
        A dictionary of dictionaries with a counts of given n-grams.

    """
    ngrams: Dict[int, Dict[Tuple[str, ...], Tensor]] = defaultdict(lambda: defaultdict(lambda: tensor(0.0)))
    for n in range(1, n_gram_order + 1):
        for ngram in (tuple(char_or_word_list[i:i + n]) for i in range(len(char_or_word_list) - n + 1)):
            ngrams[n][ngram] += tensor(1)
    return ngrams