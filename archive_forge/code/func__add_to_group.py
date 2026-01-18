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
def _add_to_group(self, group, name, button, position):
    gr = self._groups.get(group, [])
    if not gr:
        sep = self.insertSeparator(self._message_action)
        gr.append(sep)
    before = gr[position]
    widget = self.insertWidget(before, button)
    gr.insert(position, widget)
    self._groups[group] = gr