from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
class InvalidTreeError(Exception):
    """Raised when user attempts to create an invalid tree in some way."""