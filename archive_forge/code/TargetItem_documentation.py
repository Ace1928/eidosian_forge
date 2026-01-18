import string
from math import atan2
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import Symbols
from .TextItem import TextItem
from .UIGraphicsItem import UIGraphicsItem
from .ViewBox import ViewBox
Method to set how the TargetLabel should display the text.  This
        method should be called from TargetItem.setLabel directly.

        Parameters
        ----------
        text : Callable or str
            Details how to format the text.
            If Callable, then the label will display the result of ``text(x, y)``
            If a fromatted string, then the output of ``text.format(x, y)`` will be
            displayed
            If a non-formatted string, then the text label will display ``text``
        