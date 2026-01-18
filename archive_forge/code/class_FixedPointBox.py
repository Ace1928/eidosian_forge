import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
@dataclasses.dataclass
class FixedPointBox:
    value: bool = True