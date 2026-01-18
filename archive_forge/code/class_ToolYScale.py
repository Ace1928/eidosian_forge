import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
class ToolYScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the Y axis."""
    description = 'Toggle scale Y axis'
    default_keymap = property(lambda self: mpl.rcParams['keymap.yscale'])

    def set_scale(self, ax, scale):
        ax.set_yscale(scale)