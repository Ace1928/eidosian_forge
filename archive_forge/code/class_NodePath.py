from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
class NodePath(PurePosixPath):
    """Represents a path from one node to another within a tree."""

    def __init__(self, *pathsegments):
        if sys.version_info >= (3, 12):
            super().__init__(*pathsegments)
        else:
            super().__new__(PurePosixPath, *pathsegments)
        if self.drive:
            raise ValueError('NodePaths cannot have drives')
        if self.root not in ['/', '']:
            raise ValueError('Root of NodePath can only be either "/" or "", with "" meaning the path is relative.')