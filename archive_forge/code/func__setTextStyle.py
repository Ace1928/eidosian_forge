import code, sys, traceback
from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkBrush
from .CmdInput import CmdInput
def _setTextStyle(self, style):
    charFormat, blockFormat = self.textStyles[style]
    cursor = self.output.textCursor()
    cursor.setBlockFormat(blockFormat)
    self.output.setCurrentCharFormat(charFormat)