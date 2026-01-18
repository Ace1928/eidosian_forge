from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
def _update_view(self):
    """
        Update the viewlim and position from the view and position stack for
        each Axes.
        """
    nav_info = self._nav_stack()
    if nav_info is None:
        return
    items = list(nav_info.items())
    for ax, (view, (pos_orig, pos_active)) in items:
        ax._set_view(view)
        ax._set_position(pos_orig, 'original')
        ax._set_position(pos_active, 'active')
    self.canvas.draw_idle()