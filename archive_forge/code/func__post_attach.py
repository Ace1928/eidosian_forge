from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def _post_attach(self: NamedNode, parent: NamedNode) -> None:
    """Ensures child has name attribute corresponding to key under which it has been stored."""
    key = next((k for k, v in parent.children.items() if v is self))
    self.name = key