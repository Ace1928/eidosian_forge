from PySide2 import QtCore, QtGui, QtWidgets
import dragdroprobot_rc
class RobotPart(QtWidgets.QGraphicsItem):

    def __init__(self, parent=None):
        super(RobotPart, self).__init__(parent)
        self.color = QtGui.QColor(QtCore.Qt.lightGray)
        self.pixmap = None
        self.dragOver = False
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasColor() or (isinstance(self, RobotHead) and event.mimeData().hasImage()):
            event.setAccepted(True)
            self.dragOver = True
            self.update()
        else:
            event.setAccepted(False)

    def dragLeaveEvent(self, event):
        self.dragOver = False
        self.update()

    def dropEvent(self, event):
        self.dragOver = False
        if event.mimeData().hasColor():
            self.color = QtGui.QColor(event.mimeData().colorData())
        elif event.mimeData().hasImage():
            self.pixmap = QtGui.QPixmap(event.mimeData().imageData())
        self.update()