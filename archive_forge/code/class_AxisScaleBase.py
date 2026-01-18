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
class AxisScaleBase(ToolToggleBase):
    """Base Tool to toggle between linear and logarithmic."""

    def trigger(self, sender, event, data=None):
        if event.inaxes is None:
            return
        super().trigger(sender, event, data)

    def enable(self, event=None):
        self.set_scale(event.inaxes, 'log')
        self.figure.canvas.draw_idle()

    def disable(self, event=None):
        self.set_scale(event.inaxes, 'linear')
        self.figure.canvas.draw_idle()