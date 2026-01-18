import jupyter_rfb
import numpy as np
from .. import functions as fn
from .. import graphicsItems, widgets
from ..Qt import QtCore, QtGui
def addPlot(self, *args, **kwds):
    kwds['enableMenu'] = False
    plotItem = self.gfxLayout.addPlot(*args, **kwds)
    connect_viewbox_redraw(plotItem.getViewBox(), self.request_draw)
    return plotItem