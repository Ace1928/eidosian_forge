from PySide2.QtCore import Slot, Qt, QRect, QSize
from PySide2.QtGui import QColor, QPainter, QTextFormat
from PySide2.QtWidgets import QPlainTextEdit, QWidget, QTextEdit
class LineNumberArea(QWidget):

    def __init__(self, editor):
        QWidget.__init__(self, editor)
        self.codeEditor = editor

    def sizeHint(self):
        return QSize(self.codeEditor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.codeEditor.lineNumberAreaPaintEvent(event)