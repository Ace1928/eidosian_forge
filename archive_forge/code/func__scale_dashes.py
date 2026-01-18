import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def _scale_dashes(offset, dashes, lw):
    if not mpl.rcParams['lines.scale_dashes']:
        return (offset, dashes)
    scaled_offset = offset * lw
    scaled_dashes = [x * lw if x is not None else None for x in dashes] if dashes is not None else None
    return (scaled_offset, scaled_dashes)