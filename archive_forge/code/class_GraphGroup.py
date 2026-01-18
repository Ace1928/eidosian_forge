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
class GraphGroup:
    """A group in GraphBacking.

    Note: this is called a "group" to avoid name collison with "regions" in
    RVSDG and that the word "group" has less meaning as this is does not
    imply any property.
    """
    subgroups: dict[str, 'GraphGroup']
    nodes: set[str]

    @classmethod
    def make(cls):
        return cls(subgroups=defaultdict(GraphGroup.make), nodes=set())