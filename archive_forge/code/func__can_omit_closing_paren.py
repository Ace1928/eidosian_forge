import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _can_omit_closing_paren(line: Line, *, last: Leaf, line_length: int) -> bool:
    """See `can_omit_invisible_parens`."""
    length = 4 * line.depth
    seen_other_brackets = False
    for _index, leaf, leaf_length in line.enumerate_with_length():
        length += leaf_length
        if leaf is last.opening_bracket:
            if seen_other_brackets or length <= line_length:
                return True
        elif leaf.type in OPENING_BRACKETS:
            seen_other_brackets = True
    return False