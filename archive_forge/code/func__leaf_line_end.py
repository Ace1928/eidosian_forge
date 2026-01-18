import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _leaf_line_end(leaf: Leaf) -> int:
    """Returns the line number of the leaf node's last line."""
    if leaf.type == NEWLINE:
        return leaf.lineno
    else:
        return leaf.lineno + str(leaf).count('\n')