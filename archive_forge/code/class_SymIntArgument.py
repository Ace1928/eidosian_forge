from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass(repr=False)
class SymIntArgument(_Union):
    as_name: str
    as_int: int