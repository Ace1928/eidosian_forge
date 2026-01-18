import re
import math
from typing import Any, Optional, SupportsFloat, SupportsInt, Union, Type
from ..helpers import NUMERIC_INF_OR_NAN, INVALID_NUMERIC, collapse_white_spaces
from .atomic_types import AnyAtomicType
class NonPositiveInteger(Integer):
    name = 'nonPositiveInteger'
    lower_bound, higher_bound = (None, 1)