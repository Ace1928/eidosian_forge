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
def _internal_update(self, kwargs):
    """
        Update artist properties without prenormalizing them, but generating
        errors as if calling `set`.

        The lack of prenormalization is to maintain backcompatibility.
        """
    return self._update_props(kwargs, '{cls.__name__}.set() got an unexpected keyword argument {prop_name!r}')