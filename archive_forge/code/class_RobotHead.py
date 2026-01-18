from PySide2 import QtCore, QtGui, QtWidgets
import dragdroprobot_rc
class RobotHead(RobotPart):

    def boundingRect(self):
        return QtCore.QRectF(-15, -50, 30, 50)

    def paint(self, painter, option, widget=None):
        if not self.pixmap:
            painter.setBrush(self.dragOver and self.color.lighter(130) or self.color)
            painter.drawRoundedRect(-10, -30, 20, 30, 25, 25, QtCore.Qt.RelativeSize)
            painter.setBrush(QtCore.Qt.white)
            painter.drawEllipse(-7, -3 - 20, 7, 7)
            painter.drawEllipse(0, -3 - 20, 7, 7)
            painter.setBrush(QtCore.Qt.black)
            painter.drawEllipse(-5, -1 - 20, 2, 2)
            painter.drawEllipse(2, -1 - 20, 2, 2)
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawArc(-6, -2 - 20, 12, 15, 190 * 16, 160 * 16)
        else:
            painter.scale(0.2272, 0.2824)
            painter.drawPixmap(QtCore.QPointF(-15 * 4.4, -50 * 3.54), self.pixmap)