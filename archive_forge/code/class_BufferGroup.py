from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
class BufferGroup:
    """
    Due to inplace reuse an allocated buffer can have many names.
    This tracks these collections of buffers sharing underlying memory.
    """

    def __init__(self, node: ir.Buffer):
        self.node = node
        self.names = [node.get_name()]
        self.is_output = False
        self.allocation: Optional[Allocation] = None
        self.live_range = LiveRange(float('inf'), -float('inf'))

    def update_usage(self, timestep: int):
        """Expand self.live_range to include timestep"""
        self.live_range = LiveRange(min(timestep, self.live_range.begin), max(timestep, self.live_range.end))

    def sym_nbytes(self):
        return self.node.get_layout().storage_size() * self.node.get_dtype().itemsize

    def make_allocation(self):
        assert not self.allocation, 'multiple allocations'
        assert isinstance(self.live_range.begin, int), 'live ranges not computed'
        nbytes = self.sym_nbytes()
        size_hint = V.graph.sizevars.size_hint(nbytes, fallback=64)
        self.allocation = Allocation(self.node, self.live_range, size_hint=size_hint, symbolic_size=nbytes)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.names!r}, is_output={self.is_output}, live_range={self.live_range}'