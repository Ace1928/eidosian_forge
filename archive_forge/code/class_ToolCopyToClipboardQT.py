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
@backend_tools._register_tool_class(FigureCanvasQT)
class ToolCopyToClipboardQT(backend_tools.ToolCopyToClipboardBase):

    def trigger(self, *args, **kwargs):
        pixmap = self.canvas.grab()
        QtWidgets.QApplication.instance().clipboard().setPixmap(pixmap)