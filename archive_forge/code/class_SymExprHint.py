from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass(repr=False)
class SymExprHint(_Union):
    as_int: int
    as_float: float
    as_bool: bool