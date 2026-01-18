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
def _get_token_budget(self, question_tokens, max_length=None):
    """
        Computes the number of tokens left for the table after tokenizing a question, taking into account the max
        sequence length of the model.

        Args:
            question_tokens (`List[String]`):
                List of question tokens. Returns: `int`: the number of tokens left for the table, given the model max
                length.
        """
    return (max_length if max_length is not None else self.model_max_length) - self._question_encoding_cost(question_tokens)