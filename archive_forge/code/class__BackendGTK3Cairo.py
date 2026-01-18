from contextlib import nullcontext
from .backend_cairo import FigureCanvasCairo
from .backend_gtk3 import Gtk, FigureCanvasGTK3, _BackendGTK3
@_BackendGTK3.export
class _BackendGTK3Cairo(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Cairo