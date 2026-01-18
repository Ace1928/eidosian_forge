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
def _remove_backslash_line_continuation_chars(line: Line, string_indices: List[int]) -> TResult[Line]:
    """
        Merge strings that were split across multiple lines using
        line-continuation backslashes.

        Returns:
            Ok(new_line), if @line contains backslash line-continuation
            characters.
                OR
            Err(CannotTransform), otherwise.
        """
    LL = line.leaves
    indices_to_transform = []
    for string_idx in string_indices:
        string_leaf = LL[string_idx]
        if string_leaf.type == token.STRING and '\\\n' in string_leaf.value and (not has_triple_quotes(string_leaf.value)):
            indices_to_transform.append(string_idx)
    if not indices_to_transform:
        return TErr('Found no string leaves that contain backslash line continuation characters.')
    new_line = line.clone()
    new_line.comments = line.comments.copy()
    append_leaves(new_line, line, LL)
    for string_idx in indices_to_transform:
        new_string_leaf = new_line.leaves[string_idx]
        new_string_leaf.value = new_string_leaf.value.replace('\\\n', '')
    return Ok(new_line)