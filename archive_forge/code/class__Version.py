import itertools
import re
from typing import Any, Callable, NamedTuple, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
class _Version(NamedTuple):
    epoch: int
    release: Tuple[int, ...]
    dev: Optional[Tuple[str, int]]
    pre: Optional[Tuple[str, int]]
    post: Optional[Tuple[str, int]]
    local: Optional[LocalType]