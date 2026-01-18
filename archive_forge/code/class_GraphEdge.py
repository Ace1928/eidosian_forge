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
class GraphEdge:
    """An edge in GraphBacking
    """
    src: str
    dst: str
    src_port: str | None = None
    dst_port: str | None = None
    headlabel: str | None = None
    taillabel: str | None = None
    kind: str | None = None