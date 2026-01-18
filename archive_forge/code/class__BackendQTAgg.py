import ctypes
from matplotlib.transforms import Bbox
from .qt_compat import QT_API, QtCore, QtGui
from .backend_agg import FigureCanvasAgg
from .backend_qt import _BackendQT, FigureCanvasQT
from .backend_qt import (  # noqa: F401 # pylint: disable=W0611
@_BackendQT.export
class _BackendQTAgg(_BackendQT):
    FigureCanvas = FigureCanvasQTAgg