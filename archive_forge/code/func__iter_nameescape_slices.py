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
def _iter_nameescape_slices(self, string: str) -> Iterator[Tuple[Index, Index]]:
    """
        Yields:
            All ranges of @string which, if @string were to be split there,
            would result in the splitting of an \\N{...} expression (which is NOT
            allowed).
        """
    previous_was_unescaped_backslash = False
    it = iter(enumerate(string))
    for idx, c in it:
        if c == '\\':
            previous_was_unescaped_backslash = not previous_was_unescaped_backslash
            continue
        if not previous_was_unescaped_backslash or c != 'N':
            previous_was_unescaped_backslash = False
            continue
        previous_was_unescaped_backslash = False
        begin = idx - 1
        for idx, c in it:
            if c == '}':
                end = idx
                break
        else:
            raise RuntimeError(f'{self.__class__.__name__} LOGIC ERROR!')
        yield (begin, end)