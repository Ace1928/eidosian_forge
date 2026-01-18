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
def _get_last_non_comment_leaf(line: Line) -> Optional[int]:
    for leaf_idx in range(len(line.leaves) - 1, 0, -1):
        if line.leaves[leaf_idx].type != STANDALONE_COMMENT:
            return leaf_idx
    return None