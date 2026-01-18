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
def _get_all_answer_ids(self, column_ids, row_ids, answer_coordinates):
    """
        Maps answer coordinates of a question to token indexes.

        In the SQA format (TSV), the coordinates are given as (row, column) tuples. Here, we first swap them to
        (column, row) format before calling _get_all_answer_ids_from_coordinates.
        """

    def _to_coordinates(answer_coordinates_question):
        return [(coords[1], coords[0]) for coords in answer_coordinates_question]
    return self._get_all_answer_ids_from_coordinates(column_ids, row_ids, answers_list=_to_coordinates(answer_coordinates))