from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
class Hlist(List):
    """A horizontal list of boxes."""

    def __init__(self, elements: T.Sequence[Node], w: float=0.0, m: T.Literal['additional', 'exactly']='additional', do_kern: bool=True):
        super().__init__(elements)
        if do_kern:
            self.kern()
        self.hpack(w=w, m=m)

    def kern(self) -> None:
        """
        Insert `Kern` nodes between `Char` nodes to set kerning.

        The `Char` nodes themselves determine the amount of kerning they need
        (in `~Char.get_kerning`), and this function just creates the correct
        linked list.
        """
        new_children = []
        num_children = len(self.children)
        if num_children:
            for i in range(num_children):
                elem = self.children[i]
                if i < num_children - 1:
                    next = self.children[i + 1]
                else:
                    next = None
                new_children.append(elem)
                kerning_distance = elem.get_kerning(next)
                if kerning_distance != 0.0:
                    kern = Kern(kerning_distance)
                    new_children.append(kern)
            self.children = new_children

    def hpack(self, w: float=0.0, m: T.Literal['additional', 'exactly']='additional') -> None:
        """
        Compute the dimensions of the resulting boxes, and adjust the glue if
        one of those dimensions is pre-specified.  The computed sizes normally
        enclose all of the material inside the new box; but some items may
        stick out if negative glue is used, if the box is overfull, or if a
        ``\\vbox`` includes other boxes that have been shifted left.

        Parameters
        ----------
        w : float, default: 0
            A width.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose width is 'exactly' *w*; or a box
            with the natural width of the contents, plus *w* ('additional').

        Notes
        -----
        The defaults produce a box with the natural width of the contents.
        """
        h = 0.0
        d = 0.0
        x = 0.0
        total_stretch = [0.0] * 4
        total_shrink = [0.0] * 4
        for p in self.children:
            if isinstance(p, Char):
                x += p.width
                h = max(h, p.height)
                d = max(d, p.depth)
            elif isinstance(p, Box):
                x += p.width
                if not np.isinf(p.height) and (not np.isinf(p.depth)):
                    s = getattr(p, 'shift_amount', 0.0)
                    h = max(h, p.height - s)
                    d = max(d, p.depth + s)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                x += glue_spec.width
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink
            elif isinstance(p, Kern):
                x += p.width
        self.height = h
        self.depth = d
        if m == 'additional':
            w += x
        self.width = w
        x = w - x
        if x == 0.0:
            self.glue_sign = 0
            self.glue_order = 0
            self.glue_ratio = 0.0
            return
        if x > 0.0:
            self._set_glue(x, 1, total_stretch, 'Overful')
        else:
            self._set_glue(x, -1, total_shrink, 'Underful')