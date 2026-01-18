from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass(repr=False)
class OptionalTensorArgument(_Union):
    as_tensor: str
    as_none: Tuple[()]