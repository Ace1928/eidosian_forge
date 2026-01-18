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
def _get_question_tokens(self, query):
    """Tokenizes the query, taking into account the max and min question length."""
    query_tokens = self.tokenize(query)
    if self.max_question_length is not None and len(query_tokens) > self.max_question_length:
        logger.warning('Skipping query as its tokens are longer than the max question length')
        return ('', [])
    if self.min_question_length is not None and len(query_tokens) < self.min_question_length:
        logger.warning('Skipping query as its tokens are shorter than the min question length')
        return ('', [])
    return (query, query_tokens)