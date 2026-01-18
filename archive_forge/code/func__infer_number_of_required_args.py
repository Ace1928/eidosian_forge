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
def _infer_number_of_required_args(func):
    """Infer the number of required arguments for a function

    Parameters
    ----------
    func : callable
        The function that is being inspected.

    Returns
    -------
    n_args : int
        The number of required arguments of func.
    """
    argspec = inspect.getfullargspec(func)
    n_args = len(argspec.args)
    if argspec.defaults is not None:
        n_args -= len(argspec.defaults)
    return n_args