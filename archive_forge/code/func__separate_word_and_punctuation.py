from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _separate_word_and_punctuation(word: str) -> List[str]:
    """Separates out punctuations from beginning and end of words for chrF.

    Adapted from https://github.com/m-popovic/chrF and
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.

    Args:
        word: An input word to be separated from a punctuation if present.

    Return:
        A list of a single word or a separated word and punctuation.

    """
    if len(word) == 1:
        return [word]
    if word[-1] in _PUNCTUATIONS:
        return [word[:-1], word[-1]]
    if word[0] in _PUNCTUATIONS:
        return [word[0], word[1:]]
    return [word]