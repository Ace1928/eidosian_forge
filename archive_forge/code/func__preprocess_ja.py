import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def _preprocess_ja(sentence: str) -> str:
    """Preprocess japanese sentences.

    Copy from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

    Raises:
        ValueError: If input sentence is not of a type `str`.

    """
    if not isinstance(sentence, str):
        raise ValueError(f'Only strings allowed during preprocessing step, found {type(sentence)} instead')
    sentence = sentence.rstrip()
    return unicodedata.normalize('NFKC', sentence)