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
def _get_answer_ids(self, column_ids, row_ids, answer_coordinates):
    """Maps answer coordinates of a question to token indexes."""
    answer_ids, missing_count = self._get_all_answer_ids(column_ids, row_ids, answer_coordinates)
    if missing_count:
        raise ValueError("Couldn't find all answers")
    return answer_ids