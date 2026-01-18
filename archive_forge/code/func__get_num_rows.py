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
def _get_num_rows(self, table, drop_rows_to_fit):
    num_rows = table.shape[0]
    if num_rows >= self.max_row_id:
        if drop_rows_to_fit:
            num_rows = self.max_row_id - 1
        else:
            raise ValueError('Too many rows')
    return num_rows