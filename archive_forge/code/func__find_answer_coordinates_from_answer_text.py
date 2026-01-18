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
def _find_answer_coordinates_from_answer_text(self, tokenized_table, answer_text):
    """Returns all occurrences of answer_text in the table."""
    logging.info(f'answer text: {answer_text}')
    for row_index, row in enumerate(tokenized_table.rows):
        if row_index == 0:
            continue
        for col_index, cell in enumerate(row):
            token_index = self._find_tokens(cell, answer_text)
            if token_index is not None:
                yield TokenCoordinates(row_index=row_index, column_index=col_index, token_index=token_index)