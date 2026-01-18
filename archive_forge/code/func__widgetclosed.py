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
def _widgetclosed(self):
    CloseEvent('close_event', self.canvas)._process()
    if self.window._destroying:
        return
    self.window._destroying = True
    try:
        Gcf.destroy(self)
    except AttributeError:
        pass