import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def _preprocess_sentences(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], language: Literal['en', 'ja']) -> Tuple[Union[str, Sequence[str]], Sequence[Union[str, Sequence[str]]]]:
    """Preprocess strings according to language requirements.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Return:
        Tuple of lists that contain the cleaned strings for target and preds

    Raises:
        ValueError: If a different language than ``'en'`` or ``'ja'`` is used
        ValueError: If length of target not equal to length of preds
        ValueError: If objects in reference and hypothesis corpus are not strings

    """
    target, preds = _validate_inputs(hypothesis_corpus=preds, ref_corpus=target)
    if language == 'en':
        preprocess_function = _preprocess_en
    elif language == 'ja':
        preprocess_function = _preprocess_ja
    else:
        raise ValueError(f'Expected argument `language` to either be `en` or `ja` but got {language}')
    preds = [preprocess_function(pred) for pred in preds]
    target = [[preprocess_function(ref) for ref in reference] for reference in target]
    return (preds, target)