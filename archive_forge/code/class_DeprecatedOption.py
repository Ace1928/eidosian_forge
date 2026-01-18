from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
class DeprecatedOption(NamedTuple):
    key: str
    msg: str | None
    rkey: str | None
    removal_ver: str | None