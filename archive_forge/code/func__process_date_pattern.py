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
def _process_date_pattern(dp):
    """Compute a regex for each date pattern to use as a prefilter."""
    pattern, mask = dp
    regex = pattern
    regex = regex.replace('.', re.escape('.'))
    regex = regex.replace('-', re.escape('-'))
    regex = regex.replace(' ', '\\s+')
    for field, field_regex in _FIELD_TO_REGEX:
        regex = regex.replace(field, field_regex)
    assert '%' not in regex, regex
    return (pattern, mask, re.compile('^' + regex + '$'))