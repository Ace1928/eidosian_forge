from PySide2.QtCore import QFile, QIODevice, QMimeData, QPoint, Qt, QTextStream
from PySide2.QtGui import QDrag, QPalette, QPixmap
from PySide2.QtWidgets import QApplication, QFrame, QLabel, QWidget
import draggabletext_rc
class DragLabel(QLabel):

    def __init__(self, text, parent):
        super(DragLabel, self).__init__(text, parent)
        self.setAutoFillBackground(True)
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)

    def mousePressEvent(self, event):
        hotSpot = event.pos()
        mimeData = QMimeData()
        mimeData.setText(self.text())
        mimeData.setData('application/x-hotspot', b'%d %d' % (hotSpot.x(), hotSpot.y()))
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setPixmap(pixmap)
        drag.setHotSpot(hotSpot)
        dropAction = drag.exec_(Qt.CopyAction | Qt.MoveAction, Qt.CopyAction)
        if dropAction == Qt.MoveAction:
            self.close()
            self.update()