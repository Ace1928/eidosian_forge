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
def _adjust_subplots(self, axis, subaxes):
    bbox = axis.get_position()
    l, b, w, h = (bbox.x0, bbox.y0, bbox.width, bbox.height)
    if self.padding:
        width_padding = w / (1.0 / self.padding)
        height_padding = h / (1.0 / self.padding)
    else:
        width_padding, height_padding = (0, 0)
    if self.cols == 1:
        b_w = 0
    else:
        b_w = width_padding / (self.cols - 1)
    if self.rows == 1:
        b_h = 0
    else:
        b_h = height_padding / (self.rows - 1)
    ax_w = (w - (width_padding if self.cols > 1 else 0)) / self.cols
    ax_h = (h - (height_padding if self.rows > 1 else 0)) / self.rows
    r, c = (0, 0)
    for ax in subaxes.values():
        xpos = l + c * ax_w + c * b_w
        ypos = b + r * ax_h + r * b_h
        if r != self.rows - 1:
            r += 1
        else:
            r = 0
            c += 1
        if ax is not None:
            ax.set_position([xpos, ypos, ax_w, ax_h])