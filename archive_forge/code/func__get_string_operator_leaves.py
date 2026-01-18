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
def _get_string_operator_leaves(self, leaves: Iterable[Leaf]) -> List[Leaf]:
    LL = list(leaves)
    string_op_leaves = []
    i = 0
    while LL[i].type in self.STRING_OPERATORS + [token.NAME]:
        prefix_leaf = Leaf(LL[i].type, str(LL[i]).strip())
        string_op_leaves.append(prefix_leaf)
        i += 1
    return string_op_leaves