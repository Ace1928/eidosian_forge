from __future__ import annotations
from matplotlib.offsetbox import (
from matplotlib.patches import bbox_artist as mbbox_artist
from matplotlib.transforms import Affine2D, Bbox
from .patches import InsideStrokedRectangle
def _bbox_artist(*args, **kwargs):
    if DEBUG:
        mbbox_artist(*args, **kwargs)