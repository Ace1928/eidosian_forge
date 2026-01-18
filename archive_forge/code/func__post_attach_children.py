from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def _post_attach_children(self: Tree, children: Mapping[str, Tree]) -> None:
    """Method call after attaching `children`."""
    pass