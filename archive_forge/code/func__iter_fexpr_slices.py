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
def _iter_fexpr_slices(self, string: str) -> Iterator[Tuple[Index, Index]]:
    """
        Yields:
            All ranges of @string which, if @string were to be split there,
            would result in the splitting of an f-expression (which is NOT
            allowed).
        """
    if 'f' not in get_string_prefix(string).lower():
        return
    yield from iter_fexpr_spans(string)