import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _prefer_split_rhs_oop_over_rhs(rhs_oop: RHSResult, rhs: RHSResult, mode: Mode) -> bool:
    """
    Returns whether we should prefer the result from a split omitting optional parens
    (rhs_oop) over the original (rhs).
    """
    rhs_head_equal_count = [leaf.type for leaf in rhs.head.leaves].count(token.EQUAL)
    rhs_oop_head_equal_count = [leaf.type for leaf in rhs_oop.head.leaves].count(token.EQUAL)
    if rhs_head_equal_count > 1 and rhs_head_equal_count > rhs_oop_head_equal_count:
        return False
    has_closing_bracket_after_assign = False
    for leaf in reversed(rhs_oop.head.leaves):
        if leaf.type == token.EQUAL:
            break
        if leaf.type in CLOSING_BRACKETS:
            has_closing_bracket_after_assign = True
            break
    return has_closing_bracket_after_assign or (any((leaf.type == token.EQUAL for leaf in rhs_oop.head.leaves)) and is_line_short_enough(rhs_oop.head, mode=mode)) or rhs_oop.head.contains_unsplittable_type_ignore() or rhs_oop.body.contains_unsplittable_type_ignore() or rhs_oop.tail.contains_unsplittable_type_ignore()