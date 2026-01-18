from ..Qt import QtCore, QtGui, QtWidgets
class DockDrop:
    """Provides dock-dropping methods"""

    def __init__(self, dndWidget):
        self.dndWidget = dndWidget
        self.allowedAreas = {'center', 'right', 'left', 'top', 'bottom'}
        self.dndWidget.setAcceptDrops(True)
        self.dropArea = None
        self.overlay = DropAreaOverlay(dndWidget)
        self.overlay.raise_()

    def addAllowedArea(self, area):
        self.allowedAreas.update(area)

    def removeAllowedArea(self, area):
        self.allowedAreas.discard(area)

    def resizeOverlay(self, size):
        self.overlay.resize(size)

    def raiseOverlay(self):
        self.overlay.raise_()

    def dragEnterEvent(self, ev):
        src = ev.source()
        if hasattr(src, 'implements') and src.implements('dock'):
            ev.accept()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        width, height = (self.dndWidget.width(), self.dndWidget.height())
        posF = ev.posF() if hasattr(ev, 'posF') else ev.position()
        ld = posF.x()
        rd = width - ld
        td = posF.y()
        bd = height - td
        mn = min(ld, rd, td, bd)
        if mn > 30:
            self.dropArea = 'center'
        elif (ld == mn or td == mn) and mn > height / 3:
            self.dropArea = 'center'
        elif (rd == mn or ld == mn) and mn > width / 3:
            self.dropArea = 'center'
        elif rd == mn:
            self.dropArea = 'right'
        elif ld == mn:
            self.dropArea = 'left'
        elif td == mn:
            self.dropArea = 'top'
        elif bd == mn:
            self.dropArea = 'bottom'
        if ev.source() is self.dndWidget and self.dropArea == 'center':
            self.dropArea = None
            ev.ignore()
        elif self.dropArea not in self.allowedAreas:
            self.dropArea = None
            ev.ignore()
        else:
            ev.accept()
        self.overlay.setDropArea(self.dropArea)

    def dragLeaveEvent(self, ev):
        self.dropArea = None
        self.overlay.setDropArea(self.dropArea)

    def dropEvent(self, ev):
        area = self.dropArea
        if area is None:
            return
        if area == 'center':
            area = 'above'
        self.dndWidget.area.moveDock(ev.source(), area, self.dndWidget)
        self.dropArea = None
        self.overlay.setDropArea(self.dropArea)