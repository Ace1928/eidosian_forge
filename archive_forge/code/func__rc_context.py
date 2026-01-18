from contextlib import contextmanager
from itertools import chain
import matplotlib as mpl
import numpy as np
import param
from matplotlib import (
from matplotlib.font_manager import font_scalings
from mpl_toolkits.mplot3d import Axes3D  # noqa (For 3D plots)
from ...core import (
from ...core.options import SkipRendering, Store
from ...core.util import int_to_alpha, int_to_roman, wrap_tuple_streams
from ..plot import (
from ..util import attach_streams, collate, displayable
from .util import compute_ratios, fix_aspect, get_old_rcparams
@contextmanager
def _rc_context(rcparams):
    """
    Context manager that temporarily overrides the pyplot rcParams.
    """
    old_rcparams = get_old_rcparams()
    mpl.rcParams.clear()
    mpl.rcParams.update(dict(old_rcparams, **rcparams))
    try:
        yield
    finally:
        mpl.rcParams.clear()
        mpl.rcParams.update(old_rcparams)