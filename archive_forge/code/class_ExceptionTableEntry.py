import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
@dataclasses.dataclass
class ExceptionTableEntry:
    start: int
    end: int
    target: int
    depth: int
    lasti: bool