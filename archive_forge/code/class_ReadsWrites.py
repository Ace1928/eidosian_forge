import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
@dataclasses.dataclass
class ReadsWrites:
    reads: Set[Any]
    writes: Set[Any]
    visited: Set[Any]