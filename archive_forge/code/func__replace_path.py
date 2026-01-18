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
def _replace_path(self, source_class):
    """
        Changes the full path to the public API path that is used
        in sphinx. This is needed for links to work.
        """
    replace_dict = {'_base._AxesBase': 'Axes', '_axes.Axes': 'Axes'}
    for key, value in replace_dict.items():
        source_class = source_class.replace(key, value)
    return source_class