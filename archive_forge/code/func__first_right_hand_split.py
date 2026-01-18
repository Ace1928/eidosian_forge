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
def _first_right_hand_split(line: Line, omit: Collection[LeafID]=()) -> RHSResult:
    """Split the line into head, body, tail starting with the last bracket pair.

    Note: this function should not have side effects. It's relied upon by
    _maybe_split_omitting_optional_parens to get an opinion whether to prefer
    splitting on the right side of an assignment statement.
    """
    tail_leaves: List[Leaf] = []
    body_leaves: List[Leaf] = []
    head_leaves: List[Leaf] = []
    current_leaves = tail_leaves
    opening_bracket: Optional[Leaf] = None
    closing_bracket: Optional[Leaf] = None
    for leaf in reversed(line.leaves):
        if current_leaves is body_leaves:
            if leaf is opening_bracket:
                current_leaves = head_leaves if body_leaves else tail_leaves
        current_leaves.append(leaf)
        if current_leaves is tail_leaves:
            if leaf.type in CLOSING_BRACKETS and id(leaf) not in omit:
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf
                current_leaves = body_leaves
    if not (opening_bracket and closing_bracket and head_leaves):
        raise CannotSplit('No brackets found')
    tail_leaves.reverse()
    body_leaves.reverse()
    head_leaves.reverse()
    body: Optional[Line] = None
    if Preview.hug_parens_with_braces_and_square_brackets in line.mode and tail_leaves[0].value and (tail_leaves[0].opening_bracket is head_leaves[-1]):
        inner_body_leaves = list(body_leaves)
        hugged_opening_leaves: List[Leaf] = []
        hugged_closing_leaves: List[Leaf] = []
        is_unpacking = body_leaves[0].type in [token.STAR, token.DOUBLESTAR]
        unpacking_offset: int = 1 if is_unpacking else 0
        while len(inner_body_leaves) >= 2 + unpacking_offset and inner_body_leaves[-1].type in CLOSING_BRACKETS and (inner_body_leaves[-1].opening_bracket is inner_body_leaves[unpacking_offset]):
            if unpacking_offset:
                hugged_opening_leaves.append(inner_body_leaves.pop(0))
                unpacking_offset = 0
            hugged_opening_leaves.append(inner_body_leaves.pop(0))
            hugged_closing_leaves.insert(0, inner_body_leaves.pop())
        if hugged_opening_leaves and inner_body_leaves:
            inner_body = bracket_split_build_line(inner_body_leaves, line, hugged_opening_leaves[-1], component=_BracketSplitComponent.body)
            if line.mode.magic_trailing_comma and inner_body_leaves[-1].type == token.COMMA:
                should_hug = True
            else:
                line_length = line.mode.line_length - sum((len(str(leaf)) for leaf in hugged_opening_leaves + hugged_closing_leaves))
                if is_line_short_enough(inner_body, mode=replace(line.mode, line_length=line_length)):
                    should_hug = False
                else:
                    should_hug = True
            if should_hug:
                body_leaves = inner_body_leaves
                head_leaves.extend(hugged_opening_leaves)
                tail_leaves = hugged_closing_leaves + tail_leaves
                body = inner_body
    head = bracket_split_build_line(head_leaves, line, opening_bracket, component=_BracketSplitComponent.head)
    if body is None:
        body = bracket_split_build_line(body_leaves, line, opening_bracket, component=_BracketSplitComponent.body)
    tail = bracket_split_build_line(tail_leaves, line, opening_bracket, component=_BracketSplitComponent.tail)
    bracket_split_succeeded_or_raise(head, body, tail)
    return RHSResult(head, body, tail, opening_bracket, closing_bracket)