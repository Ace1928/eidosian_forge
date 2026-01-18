from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
class NotFoundInTreeError(ValueError):
    """Raised when operation can't be completed because one node is not part of the expected tree."""