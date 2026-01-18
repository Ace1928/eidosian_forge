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
@_Backend.export
class _BackendQT(_Backend):
    backend_version = __version__
    FigureCanvas = FigureCanvasQT
    FigureManager = FigureManagerQT
    mainloop = FigureManagerQT.start_main_loop