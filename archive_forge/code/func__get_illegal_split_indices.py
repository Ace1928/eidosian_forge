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
def _get_illegal_split_indices(self, string: str) -> Set[Index]:
    illegal_indices: Set[Index] = set()
    iterators = [self._iter_fexpr_slices(string), self._iter_nameescape_slices(string)]
    for it in iterators:
        for begin, end in it:
            illegal_indices.update(range(begin, end + 1))
    return illegal_indices