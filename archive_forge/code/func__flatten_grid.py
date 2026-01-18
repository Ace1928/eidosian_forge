from __future__ import annotations
import math
from collections import namedtuple
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import FlexBox as BkFlexBox, GridBox as BkGridBox
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from .base import (
@classmethod
def _flatten_grid(cls, layout, nrows=None, ncols=None):
    Item = namedtuple('Item', ['layout', 'r0', 'c0', 'r1', 'c1'])
    Grid = namedtuple('Grid', ['nrows', 'ncols', 'items'])

    def gcd(a, b):
        a, b = (abs(a), abs(b))
        while b != 0:
            a, b = (b, a % b)
        return a

    def lcm(a, *rest):
        for b in rest:
            a = a * b // gcd(a, b)
        return a
    nonempty = lambda child: child.nrows != 0 and child.ncols != 0

    def _flatten(layout, nrows=None, ncols=None):
        _flatten_ = partial(_flatten, nrows=nrows, ncols=ncols)
        if isinstance(layout, _row):
            children = list(filter(nonempty, map(_flatten_, layout.children)))
            if not children:
                return Grid(0, 0, [])
            nrows = lcm(*[child.nrows for child in children])
            if not ncols:
                ncols = sum([child.ncols for child in children])
            items = []
            offset = 0
            for child in children:
                factor = nrows // child.nrows
                for layout, r0, c0, r1, c1 in child.items:
                    items.append((layout, factor * r0, c0 + offset, factor * r1, c1 + offset))
                offset += child.ncols
            return Grid(nrows, ncols, items)
        elif isinstance(layout, _col):
            children = list(filter(nonempty, map(_flatten_, layout.children)))
            if not children:
                return Grid(0, 0, [])
            if not nrows:
                nrows = sum([child.nrows for child in children])
            ncols = lcm(*[child.ncols for child in children])
            items = []
            offset = 0
            for child in children:
                factor = ncols // child.ncols
                for layout, r0, c0, r1, c1 in child.items:
                    items.append((layout, r0 + offset, factor * c0, r1 + offset, factor * c1))
                offset += child.nrows
            return Grid(nrows, ncols, items)
        else:
            return Grid(1, 1, [Item(layout, 0, 0, 1, 1)])
    grid = _flatten(layout, nrows, ncols)
    children = []
    for layout, r0, c0, r1, c1 in grid.items:
        if layout is not None:
            children.append((layout, r0, c0, r1 - r0, c1 - c0))
    return children