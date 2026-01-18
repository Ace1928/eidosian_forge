import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
@dataclass(frozen=True)
class DifferentiableInput:
    name: str
    type: Type
    cpp_type: str