import json
import re
from collections import (
from typing import (
import attr
from . import (
from .parsing import (
def _zero_based_index(self, onebased: Union[int, str]) -> int:
    """Convert a one-based index to a zero-based index."""
    result = int(onebased)
    if result > 0:
        result -= 1
    return result