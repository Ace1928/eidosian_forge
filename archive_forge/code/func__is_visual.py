from __future__ import annotations
import logging # isort:skip
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
def _is_visual(ft):
    """Whether a feature trait name is visual"""
    feature, trait = _split_feature_trait(ft)
    return feature in ('line', 'fill', 'hatch', 'text', 'global') and trait is not None