import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
def _install_properties_docs():
    prop_doc = _parse_docs()
    for p in [member for member in dir(RegionProperties) if not member.startswith('_')]:
        getattr(RegionProperties, p).__doc__ = prop_doc[p]