from ..Qt import QtCore, QtGui, QtWidgets
class DropAreaOverlay(QtWidgets.QWidget):
    """Overlay widget that draws drop areas during a drag-drop operation"""

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.dropArea = None
        self.hide()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def setDropArea(self, area):
        self.dropArea = area
        if area is None:
            self.hide()
        else:
            prgn = self.parent().rect()
            rgn = QtCore.QRect(prgn)
            w = min(30, int(prgn.width() / 3))
            h = min(30, int(prgn.height() / 3))
            if self.dropArea == 'left':
                rgn.setWidth(w)
            elif self.dropArea == 'right':
                rgn.setLeft(rgn.left() + prgn.width() - w)
            elif self.dropArea == 'top':
                rgn.setHeight(h)
            elif self.dropArea == 'bottom':
                rgn.setTop(rgn.top() + prgn.height() - h)
            elif self.dropArea == 'center':
                rgn.adjust(w, h, -w, -h)
            self.setGeometry(rgn)
            self.show()
        self.update()

    def paintEvent(self, ev):
        if self.dropArea is None:
            return
        p = QtGui.QPainter(self)
        rgn = self.rect()
        p.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 255, 50)))
        p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 150), 3))
        p.drawRect(rgn)
        p.end()