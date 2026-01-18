from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
def add_frame(auxbox, elements):
    """
    Add frame to colorbar
    """
    from matplotlib.patches import Rectangle
    width = elements.key_width
    height = elements.key_height
    if elements.is_horizontal:
        width, height = (height, width)
    rect = Rectangle((0, 0), width, height, facecolor='none')
    auxbox.add_artist(rect)
    return rect