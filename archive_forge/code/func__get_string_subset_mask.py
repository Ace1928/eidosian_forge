from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _get_string_subset_mask(data, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Get a subset from a string array.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return names that match this regular expression.

    Returns
    -------
    data : list-like, shape<=[n_features]
        List of matching strings
    """
    mask = np.full_like(data, True, dtype=bool)
    if starts_with is not None:
        start_match = _string_vector_match(data, starts_with, lambda x, match: x.startswith(match))
        mask = np.logical_and(mask, start_match)
    if ends_with is not None:
        end_match = _string_vector_match(data, ends_with, lambda x, match: x.endswith(match))
        mask = np.logical_and(mask, end_match)
    if exact_word is not None:
        if not isinstance(exact_word, str):
            exact_word = [_exact_word_regex(w) for w in exact_word]
        else:
            exact_word = _exact_word_regex(exact_word)
        exact_word_match = _get_string_subset_mask(data, regex=exact_word)
        mask = np.logical_and(mask, exact_word_match)
    if regex is not None:
        if not isinstance(regex, str):
            regex = [re.compile(r) for r in regex]
        else:
            regex = re.compile(regex)
        regex_match = _string_vector_match(data, regex, lambda x, match: bool(match.search(x)), dtype=_re_pattern)
        mask = np.logical_and(mask, regex_match)
    return mask