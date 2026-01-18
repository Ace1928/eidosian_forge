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
def _on_value_changed(self):
    spinboxes = self._spinboxes
    for lower, higher in [('bottom', 'top'), ('left', 'right')]:
        spinboxes[higher].setMinimum(spinboxes[lower].value() + 0.001)
        spinboxes[lower].setMaximum(spinboxes[higher].value() - 0.001)
    self._figure.subplots_adjust(**{attr: spinbox.value() for attr, spinbox in spinboxes.items()})
    self._figure.canvas.draw_idle()