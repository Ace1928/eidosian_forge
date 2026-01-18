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
def _get_value_type(numeric_value):
    if numeric_value.float_value is not None:
        return NUMBER_TYPE
    elif numeric_value.date is not None:
        return DATE_TYPE
    raise ValueError(f'Unknown type: {numeric_value}')