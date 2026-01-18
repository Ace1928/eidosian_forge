import jupyter_rfb
import numpy as np
from .. import functions as fn
from .. import graphicsItems, widgets
from ..Qt import QtCore, QtGui
def addViewBox(self, *args, **kwds):
    kwds['enableMenu'] = False
    vb = self.gfxLayout.addViewBox(*args, **kwds)
    connect_viewbox_redraw(vb, self.request_draw)
    return vb