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
class Relation(enum.Enum):
    HEADER_TO_CELL = 1
    CELL_TO_HEADER = 2
    QUERY_TO_HEADER = 3
    QUERY_TO_CELL = 4
    ROW_TO_CELL = 5
    CELL_TO_ROW = 6
    EQ = 7
    LT = 8
    GT = 9