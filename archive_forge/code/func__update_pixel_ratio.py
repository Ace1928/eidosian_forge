import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
def _update_pixel_ratio(self):
    if self._set_device_pixel_ratio(self.devicePixelRatioF() or 1):
        event = QtGui.QResizeEvent(self.size(), self.size())
        self.resizeEvent(event)