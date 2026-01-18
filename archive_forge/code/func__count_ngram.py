from collections import Counter
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
def _count_ngram(ngram_input_list: Sequence[str], n_gram: int) -> Counter:
    """Count how many times each word appears in a given text with ngram.

    Args:
        ngram_input_list: A list of translated text or reference texts
        n_gram: gram value ranged 1 to 4

    Return:
        ngram_counter: a collections.Counter object of ngram

    """
    ngram_counter: Counter = Counter()
    for i in range(1, n_gram + 1):
        for j in range(len(ngram_input_list) - i + 1):
            ngram_key = tuple(ngram_input_list[j:i + j])
            ngram_counter[ngram_key] += 1
    return ngram_counter