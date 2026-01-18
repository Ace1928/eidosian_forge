import os
import re
from ...Qt import QtCore, QtGui, QtWidgets
from ..Parameter import Parameter
from .str import StrParameterItem
class FileParameterItem(StrParameterItem):

    def __init__(self, param, depth):
        self._value = None
        super().__init__(param, depth)
        button = QtWidgets.QPushButton('...')
        button.setFixedWidth(25)
        button.setContentsMargins(0, 0, 0, 0)
        button.clicked.connect(self._retrieveFileSelection_gui)
        self.layoutWidget.layout().insertWidget(2, button)
        self.displayLabel.resizeEvent = self._newResizeEvent

    def makeWidget(self):
        w = super().makeWidget()
        w.setValue = self.setValue
        w.value = self.value
        delattr(w, 'sigChanging')
        return w

    def _newResizeEvent(self, ev):
        ret = type(self.displayLabel).resizeEvent(self.displayLabel, ev)
        self.updateDisplayLabel()
        return ret

    def setValue(self, value):
        self._value = value
        self.widget.setText(str(value))

    def value(self):
        return self._value

    def _retrieveFileSelection_gui(self):
        curVal = self.param.value() if self.param.hasValue() else None
        if isinstance(curVal, list) and len(curVal):
            curVal = curVal[0]
            if os.path.isfile(curVal):
                curVal = os.path.dirname(curVal)
        opts = self.param.opts.copy()
        useDir = curVal or opts.get('directory') or os.getcwd()
        startDir = os.path.abspath(useDir)
        if os.path.isfile(startDir):
            opts['selectFile'] = os.path.basename(startDir)
            startDir = os.path.dirname(startDir)
        if os.path.exists(startDir):
            opts['directory'] = startDir
        if opts.get('windowTitle') is None:
            opts['windowTitle'] = self.param.title()
        if (fname := popupFilePicker(None, **opts)):
            self.param.setValue(fname)

    def updateDefaultBtn(self):
        self.defaultBtn.setEnabled(not self.param.valueIsDefault() and self.param.opts['enabled'])
        self.defaultBtn.setVisible(self.param.hasDefault())

    def updateDisplayLabel(self, value=None):
        lbl = self.displayLabel
        if value is None:
            value = self.param.value()
        value = str(value)
        font = lbl.font()
        metrics = QtGui.QFontMetricsF(font)
        value = metrics.elidedText(value, QtCore.Qt.TextElideMode.ElideLeft, lbl.width() - 5)
        return super().updateDisplayLabel(value)