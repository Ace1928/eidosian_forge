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
def _serialize_text(self, question_tokens):
    """Serializes texts in index arrays."""
    tokens = []
    segment_ids = []
    column_ids = []
    row_ids = []
    tokens.append(self.cls_token)
    segment_ids.append(0)
    column_ids.append(0)
    row_ids.append(0)
    for token in question_tokens:
        tokens.append(token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)
    return (tokens, segment_ids, column_ids, row_ids)