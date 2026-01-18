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
def _get_value_as_primitive_value(numeric_value):
    """Maps a NumericValue proto to a float or tuple of float."""
    if numeric_value.float_value is not None:
        return numeric_value.float_value
    if numeric_value.date is not None:
        date = numeric_value.date
        value_tuple = [None, None, None]
        if date.year is not None:
            value_tuple[0] = float(date.year)
        if date.month is not None:
            value_tuple[1] = float(date.month)
        if date.day is not None:
            value_tuple[2] = float(date.day)
        return tuple(value_tuple)
    raise ValueError(f'Unknown type: {numeric_value}')