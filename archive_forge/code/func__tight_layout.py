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
def _tight_layout(self):
    self._figure.tight_layout()
    for attr, spinbox in self._spinboxes.items():
        spinbox.blockSignals(True)
        spinbox.setValue(getattr(self._figure.subplotpars, attr))
        spinbox.blockSignals(False)
    self._figure.canvas.draw_idle()