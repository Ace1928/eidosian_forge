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
def _merge_string_group(self, line: Line, string_indices: List[int]) -> TResult[Line]:
    """
        Merges string groups (i.e. set of adjacent strings).

        Each index from `string_indices` designates one string group's first
        leaf in `line.leaves`.

        Returns:
            Ok(new_line), if ALL of the validation checks found in
            _validate_msg(...) pass.
                OR
            Err(CannotTransform), otherwise.
        """
    LL = line.leaves
    is_valid_index = is_valid_index_factory(LL)
    merged_string_idx_dict: Dict[int, Tuple[int, Leaf]] = {}
    for string_idx in string_indices:
        vresult = self._validate_msg(line, string_idx)
        if isinstance(vresult, Err):
            continue
        merged_string_idx_dict[string_idx] = self._merge_one_string_group(LL, string_idx, is_valid_index)
    if not merged_string_idx_dict:
        return TErr('No string group is merged')
    new_line = line.clone()
    previous_merged_string_idx = -1
    previous_merged_num_of_strings = -1
    for i, leaf in enumerate(LL):
        if i in merged_string_idx_dict:
            previous_merged_string_idx = i
            previous_merged_num_of_strings, string_leaf = merged_string_idx_dict[i]
            new_line.append(string_leaf)
        if previous_merged_string_idx <= i < previous_merged_string_idx + previous_merged_num_of_strings:
            for comment_leaf in line.comments_after(LL[i]):
                new_line.append(comment_leaf, preformatted=True)
            continue
        append_leaves(new_line, line, [leaf])
    return Ok(new_line)