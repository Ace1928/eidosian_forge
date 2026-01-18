from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def _recursively_make_subfig_transparent(exit_stack, subfig):
    exit_stack.enter_context(subfig.patch._cm_set(facecolor='none', edgecolor='none'))
    for ax in subfig.axes:
        exit_stack.enter_context(ax.patch._cm_set(facecolor='none', edgecolor='none'))
    for sub_subfig in subfig.subfigs:
        _recursively_make_subfig_transparent(exit_stack, sub_subfig)