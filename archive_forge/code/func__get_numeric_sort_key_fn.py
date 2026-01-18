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
def _get_numeric_sort_key_fn(self, table_numeric_values, value):
    """
        Returns the sort key function for comparing value to table values. The function returned will be a suitable
        input for the key param of the sort(). See number_annotation_utils._get_numeric_sort_key_fn for details

        Args:
            table_numeric_values: Numeric values of a column
            value: Numeric value in the question

        Returns:
            A function key function to compare column and question values.
        """
    if not table_numeric_values:
        return None
    all_values = list(table_numeric_values.values())
    all_values.append(value)
    try:
        return get_numeric_sort_key_fn(all_values)
    except ValueError:
        return None