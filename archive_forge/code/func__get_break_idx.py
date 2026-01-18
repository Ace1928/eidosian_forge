import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _get_break_idx(self, string: str, max_break_idx: int) -> Optional[int]:
    """
        This method contains the algorithm that StringSplitter uses to
        determine which character to split each string at.

        Args:
            @string: The substring that we are attempting to split.
            @max_break_idx: The ideal break index. We will return this value if it
            meets all the necessary conditions. In the likely event that it
            doesn't we will try to find the closest index BELOW @max_break_idx
            that does. If that fails, we will expand our search by also
            considering all valid indices ABOVE @max_break_idx.

        Pre-Conditions:
            * assert_is_leaf_string(@string)
            * 0 <= @max_break_idx < len(@string)

        Returns:
            break_idx, if an index is able to be found that meets all of the
            conditions listed in the 'Transformations' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
    is_valid_index = is_valid_index_factory(string)
    assert is_valid_index(max_break_idx)
    assert_is_leaf_string(string)
    _illegal_split_indices = self._get_illegal_split_indices(string)

    def breaks_unsplittable_expression(i: Index) -> bool:
        """
            Returns:
                True iff returning @i would result in the splitting of an
                unsplittable expression (which is NOT allowed).
            """
        return i in _illegal_split_indices

    def passes_all_checks(i: Index) -> bool:
        """
            Returns:
                True iff ALL of the conditions listed in the 'Transformations'
                section of this classes' docstring would be met by returning @i.
            """
        is_space = string[i] == ' '
        is_split_safe = is_valid_index(i - 1) and string[i - 1] in SPLIT_SAFE_CHARS
        is_not_escaped = True
        j = i - 1
        while is_valid_index(j) and string[j] == '\\':
            is_not_escaped = not is_not_escaped
            j -= 1
        is_big_enough = len(string[i:]) >= self.MIN_SUBSTR_SIZE and len(string[:i]) >= self.MIN_SUBSTR_SIZE
        return (is_space or is_split_safe) and is_not_escaped and is_big_enough and (not breaks_unsplittable_expression(i))
    break_idx = max_break_idx
    while is_valid_index(break_idx - 1) and (not passes_all_checks(break_idx)):
        break_idx -= 1
    if not passes_all_checks(break_idx):
        break_idx = max_break_idx + 1
        while is_valid_index(break_idx + 1) and (not passes_all_checks(break_idx)):
            break_idx += 1
        if not is_valid_index(break_idx) or not passes_all_checks(break_idx):
            return None
    return break_idx