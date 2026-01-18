from __future__ import annotations
import typing
import numpy as np
from mizani.bounds import expand_range_distinct
from .._utils import ignore_warnings
from ..iapi import range_view

    Expand Coordinate Range in coordinate space

    Parameters
    ----------
    x:
        (max, min) in data scale
    expand:
        How to expand
    trans:
        Coordinate transformation
    