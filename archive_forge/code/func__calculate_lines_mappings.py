import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _calculate_lines_mappings(original_source: str, modified_source: str) -> Sequence[_LinesMapping]:
    """Returns a sequence of _LinesMapping by diffing the sources.

    For example, given the following diff:
        import re
      - def func(arg1,
      -   arg2, arg3):
      + def func(arg1, arg2, arg3):
          pass
    It returns the following mappings:
      original -> modified
       (1, 1)  ->  (1, 1), is_changed_block=False (the "import re" line)
       (2, 3)  ->  (2, 2), is_changed_block=True (the diff)
       (4, 4)  ->  (3, 3), is_changed_block=False (the "pass" line)

    You can think of this visually as if it brings up a side-by-side diff, and tries
    to map the line ranges from the left side to the right side:

      (1, 1)->(1, 1)    1. import re          1. import re
      (2, 3)->(2, 2)    2. def func(arg1,     2. def func(arg1, arg2, arg3):
                        3.   arg2, arg3):
      (4, 4)->(3, 3)    4.   pass             3.   pass

    Args:
      original_source: the original source.
      modified_source: the modified source.
    """
    matcher = difflib.SequenceMatcher(None, original_source.splitlines(keepends=True), modified_source.splitlines(keepends=True))
    matching_blocks = matcher.get_matching_blocks()
    lines_mappings: List[_LinesMapping] = []
    for i, block in enumerate(matching_blocks):
        if i == 0:
            if block.a != 0 or block.b != 0:
                lines_mappings.append(_LinesMapping(original_start=1, original_end=block.a, modified_start=1, modified_end=block.b, is_changed_block=False))
        else:
            previous_block = matching_blocks[i - 1]
            lines_mappings.append(_LinesMapping(original_start=previous_block.a + previous_block.size + 1, original_end=block.a, modified_start=previous_block.b + previous_block.size + 1, modified_end=block.b, is_changed_block=True))
        if i < len(matching_blocks) - 1:
            lines_mappings.append(_LinesMapping(original_start=block.a + 1, original_end=block.a + block.size, modified_start=block.b + 1, modified_end=block.b + block.size, is_changed_block=False))
    return lines_mappings