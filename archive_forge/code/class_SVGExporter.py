import contextlib
import re
import xml.dom.minidom as xml
import numpy as np
from .. import debug
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtSvg, QtWidgets
from .Exporter import Exporter
class SVGExporter(Exporter):
    Name = 'Scalable Vector Graphics (SVG)'
    allowCopy = True

    def __init__(self, item):
        Exporter.__init__(self, item)
        tr = self.getTargetRect()
        scene = item.scene() if isinstance(item, QtWidgets.QGraphicsItem) else item
        bgbrush = scene.views()[0].backgroundBrush()
        bg = bgbrush.color()
        if bgbrush.style() == QtCore.Qt.BrushStyle.NoBrush:
            bg.setAlpha(0)
        self.params = Parameter.create(name='params', type='group', children=[{'name': 'background', 'title': translate('Exporter', 'background'), 'type': 'color', 'value': bg}, {'name': 'width', 'title': translate('Exporter', 'width'), 'type': 'float', 'value': tr.width(), 'limits': (0, None)}, {'name': 'height', 'title': translate('Exporter', 'height'), 'type': 'float', 'value': tr.height(), 'limits': (0, None)}, {'name': 'scaling stroke', 'title': translate('Exporter', 'scaling stroke'), 'type': 'bool', 'value': False, 'tip': 'If False, strokes are non-scaling, which means that they appear the same width on screen regardless of how they are scaled or how the view is zoomed.'}])
        self.params.param('width').sigValueChanged.connect(self.widthChanged)
        self.params.param('height').sigValueChanged.connect(self.heightChanged)

    def widthChanged(self):
        sr = self.getSourceRect()
        ar = sr.height() / sr.width()
        self.params.param('height').setValue(self.params['width'] * ar, blockSignal=self.heightChanged)

    def heightChanged(self):
        sr = self.getSourceRect()
        ar = sr.width() / sr.height()
        self.params.param('width').setValue(self.params['height'] * ar, blockSignal=self.widthChanged)

    def parameters(self):
        return self.params

    def export(self, fileName=None, toBytes=False, copy=False):
        if toBytes is False and copy is False and (fileName is None):
            self.fileSaveDialog(filter=f'{translate('Exporter', 'Scalable Vector Graphics')} (*.svg)')
            return
        options = {ch.name(): ch.value() for ch in self.params.children()}
        options['background'] = self.params['background']
        options['width'] = self.params['width']
        options['height'] = self.params['height']
        xml = generateSvg(self.item, options)
        if toBytes:
            return xml.encode('UTF-8')
        elif copy:
            md = QtCore.QMimeData()
            md.setData('image/svg+xml', QtCore.QByteArray(xml.encode('UTF-8')))
            QtWidgets.QApplication.clipboard().setMimeData(md)
        else:
            with open(fileName, 'wb') as fh:
                fh.write(xml.encode('utf-8'))