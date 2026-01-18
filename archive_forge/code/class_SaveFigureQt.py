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
class SaveFigureQt(backend_tools.SaveFigureBase):

    def trigger(self, *args):
        NavigationToolbar2QT.save_figure(self._make_classic_style_pseudo_toolbar())