import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
def addGradientListToDocstring():
    """Decorator to add list of current pre-defined gradients to the end of a function docstring."""

    def dec(fn):
        if fn.__doc__ is not None:
            fn.__doc__ = fn.__doc__ + str(list(Gradients.keys())).strip('[').strip(']')
        return fn
    return dec