import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
class ColorMapDisplayMixin:

    def __init__(self, *, orientation):
        self.horizontal = orientation == 'horizontal'
        self._menu = None
        self._setColorMap(None)

    def setMaximumThickness(self, val):
        Thickness = 'Height' if self.horizontal else 'Width'
        getattr(self, f'setMaximum{Thickness}')(val)

    def _setColorMap(self, cmap):
        if isinstance(cmap, str):
            try:
                cmap = colormap.get(cmap)
            except FileNotFoundError:
                cmap = None
        if cmap is None:
            cmap = colormap.ColorMap(None, [0.0, 1.0])
        self._cmap = cmap
        self._image = None

    def setColorMap(self, cmap):
        self._setColorMap(cmap)
        self.colorMapChanged()

    def colorMap(self):
        return self._cmap

    def getImage(self):
        if self._image is None:
            lut = self._cmap.getLookupTable(nPts=256, alpha=True)
            lut = np.expand_dims(lut, axis=0 if self.horizontal else 1)
            qimg = fn.ndarray_to_qimage(lut, QtGui.QImage.Format.Format_RGBA8888)
            self._image = qimg if self.horizontal else qimg.mirrored()
        return self._image

    def getMenu(self):
        if self._menu is None:
            self._menu = ColorMapMenu()
            self._menu.triggered.connect(self.menuTriggered)
        return self._menu

    def menuTriggered(self, action):
        name, source = action.data()
        if name is None:
            cmap = None
        elif source == 'preset-gradient':
            cmap = preset_gradient_to_colormap(name)
        else:
            cmap = colormap.get(name, source=source)
        self.setColorMap(cmap)

    def paintColorMap(self, painter, rect):
        painter.save()
        image = self.getImage()
        painter.drawImage(rect, image)
        if not self.horizontal:
            painter.translate(rect.center())
            painter.rotate(-90)
            painter.translate(-rect.center())
        text = self.colorMap().name
        wpen = QtGui.QPen(QtCore.Qt.GlobalColor.white)
        bpen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
        lightness = image.pixelColor(image.rect().center()).lightnessF()
        if lightness >= 0.5:
            pens = [wpen, bpen]
        else:
            pens = [bpen, wpen]
        AF = QtCore.Qt.AlignmentFlag
        trect = painter.boundingRect(rect, AF.AlignCenter, text)
        painter.setPen(pens[0])
        painter.drawText(trect, 0, text)
        painter.setPen(pens[1])
        painter.drawText(trect.adjusted(1, 0, 1, 0), 0, text)
        painter.restore()