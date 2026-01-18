from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass(repr=False)
class SymBool(_Union):
    as_expr: SymExpr
    as_bool: bool