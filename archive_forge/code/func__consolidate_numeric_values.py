import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
def _consolidate_numeric_values(row_index_to_values, min_consolidation_fraction, debug_info):
    """
    Finds the most common numeric values in a column and returns them

    Args:
        row_index_to_values:
            For each row index all the values in that cell.
        min_consolidation_fraction:
            Fraction of cells that need to have consolidated value.
        debug_info:
            Additional information only used for logging

    Returns:
        For each row index the first value that matches the most common value. Rows that don't have a matching value
        are dropped. Empty list if values can't be consolidated.
    """
    type_counts = collections.Counter()
    for numeric_values in row_index_to_values.values():
        type_counts.update(_get_all_types(numeric_values))
    if not type_counts:
        return {}
    max_count = max(type_counts.values())
    if max_count < len(row_index_to_values) * min_consolidation_fraction:
        return {}
    valid_types = set()
    for value_type, count in type_counts.items():
        if count == max_count:
            valid_types.add(value_type)
    if len(valid_types) > 1:
        assert DATE_TYPE in valid_types
        max_type = DATE_TYPE
    else:
        max_type = next(iter(valid_types))
    new_row_index_to_value = {}
    for index, values in row_index_to_values.items():
        for value in values:
            if _get_value_type(value) == max_type:
                new_row_index_to_value[index] = value
                break
    return new_row_index_to_value