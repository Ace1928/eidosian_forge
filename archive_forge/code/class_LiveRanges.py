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
class LiveRanges:
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """

    def __init__(self, ranges: Iterable[LiveRange]):
        ranges = [*sorted(ranges, key=lambda x: x.begin)]
        self.ranges = ranges[:1]
        for r in ranges[1:]:
            assert self.ranges[-1].begin <= r.begin
            if self.ranges[-1].end >= r.begin:
                self.ranges[-1] = LiveRange.join(self.ranges[-1], r)
            else:
                self.ranges.append(r)

    def overlaps(self, other: LiveRanges):
        """Check if any pair of ranges in self and other overlap"""
        left = collections.deque(self.ranges)
        right = collections.deque(other.ranges)
        while left and right:
            if left[0].begin > right[0].begin:
                left, right = (right, left)
            assert left[0].begin <= right[0].begin
            if left[0].end > right[0].begin:
                return True
            left.popleft()
        return False

    @property
    def begin(self):
        return self.ranges[0].begin

    @property
    def end(self):
        return self.ranges[-1].end

    def __repr__(self):
        return f'{self.__class__.__name__}([{', '.join(map(repr, self.ranges))}])'