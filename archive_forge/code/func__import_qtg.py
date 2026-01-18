from __future__ import division
import numpy as np
from pygsp import utils
def _import_qtg():
    try:
        import pyqtgraph as qtg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtGui
    except Exception:
        raise ImportError('Cannot import pyqtgraph. Choose another backend or try to install it with pip (or conda) install pyqtgraph. You will also need PyQt5 (or PySide) and PyOpenGL.')
    return (qtg, gl, QtGui)