from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def _finalize_rasterization(draw):
    """
    Decorator for Artist.draw method. Needed on the outermost artist, i.e.
    Figure, to finish up if the render is still in rasterized mode.
    """

    @wraps(draw)
    def draw_wrapper(artist, renderer, *args, **kwargs):
        result = draw(artist, renderer, *args, **kwargs)
        if renderer._rasterizing:
            renderer.stop_rasterizing()
            renderer._rasterizing = False
        return result
    return draw_wrapper