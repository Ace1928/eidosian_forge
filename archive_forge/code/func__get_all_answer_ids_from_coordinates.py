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
def _get_all_answer_ids_from_coordinates(self, column_ids, row_ids, answers_list):
    """Maps lists of answer coordinates to token indexes."""
    answer_ids = [0] * len(column_ids)
    found_answers = set()
    all_answers = set()
    for answers in answers_list:
        column_index, row_index = answers
        all_answers.add((column_index, row_index))
        for index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
            found_answers.add((column_index, row_index))
            answer_ids[index] = 1
    missing_count = len(all_answers) - len(found_answers)
    return (answer_ids, missing_count)