from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def _get_level_lengths(index: Index, sparsify: bool, max_index: int, hidden_elements: Sequence[int] | None=None):
    """
    Given an index, find the level length for each element.

    Parameters
    ----------
    index : Index
        Index or columns to determine lengths of each element
    sparsify : bool
        Whether to hide or show each distinct element in a MultiIndex
    max_index : int
        The maximum number of elements to analyse along the index due to trimming
    hidden_elements : sequence of int
        Index positions of elements hidden from display in the index affecting
        length

    Returns
    -------
    Dict :
        Result is a dictionary of (level, initial_position): span
    """
    if isinstance(index, MultiIndex):
        levels = index._format_multi(sparsify=lib.no_default, include_names=False)
    else:
        levels = index._format_flat(include_name=False)
    if hidden_elements is None:
        hidden_elements = []
    lengths = {}
    if not isinstance(index, MultiIndex):
        for i, value in enumerate(levels):
            if i not in hidden_elements:
                lengths[0, i] = 1
        return lengths
    for i, lvl in enumerate(levels):
        visible_row_count = 0
        for j, row in enumerate(lvl):
            if visible_row_count > max_index:
                break
            if not sparsify:
                if j not in hidden_elements:
                    lengths[i, j] = 1
                    visible_row_count += 1
            elif row is not lib.no_default and j not in hidden_elements:
                last_label = j
                lengths[i, last_label] = 1
                visible_row_count += 1
            elif row is not lib.no_default:
                last_label = j
                lengths[i, last_label] = 0
            elif j not in hidden_elements:
                visible_row_count += 1
                if visible_row_count > max_index:
                    break
                if lengths[i, last_label] == 0:
                    last_label = j
                    lengths[i, last_label] = 1
                else:
                    lengths[i, last_label] += 1
    non_zero_lengths = {element: length for element, length in lengths.items() if length >= 1}
    return non_zero_lengths