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
def _get_max_num_tokens(self, question_tokens, tokenized_table, num_columns, num_rows, max_length):
    """Computes max number of tokens that can be squeezed into the budget."""
    token_budget = self._get_token_budget(question_tokens, max_length)
    _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
    if self.cell_trim_length >= 0 and max_num_tokens > self.cell_trim_length:
        max_num_tokens = self.cell_trim_length
    num_tokens = 0
    for num_tokens in range(max_num_tokens + 1):
        cost = self._get_table_cost(tokenized_table, num_columns, num_rows, num_tokens + 1)
        if cost > token_budget:
            break
    if num_tokens < max_num_tokens:
        if self.cell_trim_length >= 0:
            return None
        if num_tokens == 0:
            return None
    return num_tokens