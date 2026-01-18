import sys
import numpy as np
from . import _backend_tk
from .backend_cairo import cairo, FigureCanvasCairo
from ._backend_tk import _BackendTk, FigureCanvasTk
@_BackendTk.export
class _BackendTkCairo(_BackendTk):
    FigureCanvas = FigureCanvasTkCairo