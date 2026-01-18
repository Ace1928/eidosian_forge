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
def _tokenize_table(self, table=None):
    """
        Tokenizes column headers and cell texts of a table.

        Args:
            table (`pd.Dataframe`):
                Table. Returns: `TokenizedTable`: TokenizedTable object.
        """
    tokenized_rows = []
    tokenized_row = []
    for column in table:
        if self.strip_column_names:
            tokenized_row.append(self.tokenize(''))
        else:
            tokenized_row.append(self.tokenize(column))
    tokenized_rows.append(tokenized_row)
    for idx, row in table.iterrows():
        tokenized_row = []
        for cell in row:
            tokenized_row.append(self.tokenize(cell))
        tokenized_rows.append(tokenized_row)
    token_coordinates = []
    for row_index, row in enumerate(tokenized_rows):
        for column_index, cell in enumerate(row):
            for token_index, _ in enumerate(cell):
                token_coordinates.append(TokenCoordinates(row_index=row_index, column_index=column_index, token_index=token_index))
    return TokenizedTable(rows=tokenized_rows, selected_tokens=token_coordinates)