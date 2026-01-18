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
def add_custom_splits(self, string: str, custom_splits: Iterable[CustomSplit]) -> None:
    """Custom Split Map Setter Method

        Side Effects:
            Adds a mapping from @string to the custom splits @custom_splits.
        """
    key = self._get_key(string)
    self._CUSTOM_SPLIT_MAP[key] = tuple(custom_splits)