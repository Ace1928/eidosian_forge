import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
@dataclass(frozen=True)
class GraphNode:
    """A node in GraphBacking
    """
    kind: str
    parent_regions: tuple[str, ...] = ()
    ports: tuple[str, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)