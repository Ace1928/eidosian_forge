from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _get_words_and_punctuation(sentence: str) -> List[str]:
    """Separates out punctuations from beginning and end of words for chrF for all words in the sentence.

    Args:
        sentence: An input sentence to split

    Return:
        An aggregated list of separated words and punctuations.

    """
    return list(chain.from_iterable((_separate_word_and_punctuation(word) for word in sentence.strip().split())))