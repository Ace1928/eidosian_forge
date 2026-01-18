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
@staticmethod
def _prefer_paren_wrap_match(LL: List[Leaf]) -> Optional[int]:
    """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the "prefer paren wrap" statement
            requirements listed in the 'Requirements' section of the StringParenWrapper
            class's docstring.
                OR
            None, otherwise.
        """
    if LL[0].type != token.STRING:
        return None
    matching_nodes = [syms.listmaker, syms.dictsetmaker, syms.testlist_gexp]
    if parent_type(LL[0]) in matching_nodes or parent_type(LL[0].parent) in matching_nodes:
        prev_sibling = LL[0].prev_sibling
        next_sibling = LL[0].next_sibling
        if not prev_sibling and (not next_sibling) and (parent_type(LL[0]) == syms.atom):
            parent = LL[0].parent
            assert parent is not None
            prev_sibling = parent.prev_sibling
            next_sibling = parent.next_sibling
        if (not prev_sibling or prev_sibling.type == token.COMMA) and (not next_sibling or next_sibling.type == token.COMMA):
            return 0
    return None