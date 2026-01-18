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
def _set_gc_clip(self, gc):
    """Set the clip properly for the gc."""
    if self._clipon:
        if self.clipbox is not None:
            gc.set_clip_rectangle(self.clipbox)
        gc.set_clip_path(self._clippath)
    else:
        gc.set_clip_rectangle(None)
        gc.set_clip_path(None)