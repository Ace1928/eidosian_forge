from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def _pre_attach(self: Tree, parent: Tree) -> None:
    """Method call before attaching to `parent`."""
    pass