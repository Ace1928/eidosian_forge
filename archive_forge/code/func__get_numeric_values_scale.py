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
def _get_numeric_values_scale(self, table, column_ids, row_ids):
    """Returns a scale to each token to down weigh the value of long words."""
    numeric_values_scale = [1.0] * len(column_ids)
    if table is None:
        return numeric_values_scale
    num_rows = table.shape[0]
    num_columns = table.shape[1]
    for col_index in range(num_columns):
        for row_index in range(num_rows):
            indices = list(self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index))
            num_indices = len(indices)
            if num_indices > 1:
                for index in indices:
                    numeric_values_scale[index] = float(num_indices)
    return numeric_values_scale