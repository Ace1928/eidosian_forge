from __future__ import annotations
from typing import (
import pytest
import numpy as np
import numpy.typing as npt
import numpy._typing as _npt
class TypeTup(NamedTuple):
    typ: type
    args: tuple[type, ...]
    origin: None | type