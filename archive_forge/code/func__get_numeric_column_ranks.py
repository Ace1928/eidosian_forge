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
def _get_numeric_column_ranks(self, column_ids, row_ids, table):
    """Returns column ranks for all numeric columns."""
    ranks = [0] * len(column_ids)
    inv_ranks = [0] * len(column_ids)
    if table is not None:
        for col_index in range(len(table.columns)):
            table_numeric_values = self._get_column_values(table, col_index)
            if not table_numeric_values:
                continue
            try:
                key_fn = get_numeric_sort_key_fn(table_numeric_values.values())
            except ValueError:
                continue
            table_numeric_values = {row_index: key_fn(value) for row_index, value in table_numeric_values.items()}
            table_numeric_values_inv = collections.defaultdict(list)
            for row_index, value in table_numeric_values.items():
                table_numeric_values_inv[value].append(row_index)
            unique_values = sorted(table_numeric_values_inv.keys())
            for rank, value in enumerate(unique_values):
                for row_index in table_numeric_values_inv[value]:
                    for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                        ranks[index] = rank + 1
                        inv_ranks[index] = len(unique_values) - rank
    return (ranks, inv_ranks)