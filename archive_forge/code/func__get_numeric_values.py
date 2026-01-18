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
def _get_numeric_values(self, table, column_ids, row_ids):
    """Returns numeric values for computation of answer loss."""
    numeric_values = [float('nan')] * len(column_ids)
    if table is not None:
        num_rows = table.shape[0]
        num_columns = table.shape[1]
        for col_index in range(num_columns):
            for row_index in range(num_rows):
                numeric_value = table.iloc[row_index, col_index].numeric_value
                if numeric_value is not None:
                    if numeric_value.float_value is None:
                        continue
                    float_value = numeric_value.float_value
                    if float_value == float('inf'):
                        continue
                    for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                        numeric_values[index] = float_value
    return numeric_values