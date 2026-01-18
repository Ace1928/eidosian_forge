from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _char_and_word_ngrams_counts(sentence: str, n_char_order: int, n_word_order: int, lowercase: bool) -> Tuple[Dict[int, Dict[Tuple[str, ...], Tensor]], Dict[int, Dict[Tuple[str, ...], Tensor]]]:
    """Get a dictionary of dictionaries with a counts of given n-grams."""
    if lowercase:
        sentence = sentence.lower()
    char_n_grams_counts = _ngram_counts(_get_characters(sentence, whitespace), n_char_order)
    word_n_grams_counts = _ngram_counts(_get_words_and_punctuation(sentence), n_word_order)
    return (char_n_grams_counts, word_n_grams_counts)