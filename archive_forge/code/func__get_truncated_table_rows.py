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
def _get_truncated_table_rows(self, query_tokens: List[str], tokenized_table: TokenizedTable, num_rows: int, num_columns: int, max_length: int, truncation_strategy: Union[str, TapasTruncationStrategy]) -> Tuple[int, int]:
    """
        Truncates a sequence pair in-place following the strategy.

        Args:
            query_tokens (`List[str]`):
                List of strings corresponding to the tokenized query.
            tokenized_table (`TokenizedTable`):
                Tokenized table
            num_rows (`int`):
                Total number of table rows
            num_columns (`int`):
                Total number of table columns
            max_length (`int`):
                Total maximum length.
            truncation_strategy (`str` or [`TapasTruncationStrategy`]):
                Truncation strategy to use. Seeing as this method should only be called when truncating, the only
                available strategy is the `"drop_rows_to_fit"` strategy.

        Returns:
            `Tuple(int, int)`: tuple containing the number of rows after truncation, and the number of tokens available
            for each table element.
        """
    if not isinstance(truncation_strategy, TapasTruncationStrategy):
        truncation_strategy = TapasTruncationStrategy(truncation_strategy)
    if max_length is None:
        max_length = self.model_max_length
    if truncation_strategy == TapasTruncationStrategy.DROP_ROWS_TO_FIT:
        while True:
            num_tokens = self._get_max_num_tokens(query_tokens, tokenized_table, num_rows=num_rows, num_columns=num_columns, max_length=max_length)
            if num_tokens is not None:
                break
            num_rows -= 1
            if num_rows < 1:
                break
    elif truncation_strategy != TapasTruncationStrategy.DO_NOT_TRUNCATE:
        raise ValueError(f'Unknown truncation strategy {truncation_strategy}.')
    return (num_rows, num_tokens or 1)