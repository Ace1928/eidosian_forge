import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
class DockLabel(VerticalLabel):
    sigClicked = QtCore.Signal(object, object)
    sigCloseClicked = QtCore.Signal()

    def __init__(self, text, closable=False, fontSize='12px'):
        self.dim = False
        self.fixedWidth = False
        self.fontSize = fontSize
        VerticalLabel.__init__(self, text, orientation='horizontal', forceWidth=False)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.dock = None
        self.updateStyle()
        self.setAutoFillBackground(False)
        self.mouseMoved = False
        self.closeButton = None
        if closable:
            self.closeButton = QtWidgets.QToolButton(self)
            self.closeButton.clicked.connect(self.sigCloseClicked)
            self.closeButton.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton))

    def updateStyle(self):
        r = '3px'
        if self.dim:
            fg = '#aaa'
            bg = '#44a'
            border = '#339'
        else:
            fg = '#fff'
            bg = '#66c'
            border = '#55B'
        if self.orientation == 'vertical':
            self.vStyle = 'DockLabel {\n                background-color : %s;\n                color : %s;\n                border-top-right-radius: 0px;\n                border-top-left-radius: %s;\n                border-bottom-right-radius: 0px;\n                border-bottom-left-radius: %s;\n                border-width: 0px;\n                border-right: 2px solid %s;\n                padding-top: 3px;\n                padding-bottom: 3px;\n                font-size: %s;\n            }' % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = 'DockLabel {\n                background-color : %s;\n                color : %s;\n                border-top-right-radius: %s;\n                border-top-left-radius: %s;\n                border-bottom-right-radius: 0px;\n                border-bottom-left-radius: 0px;\n                border-width: 0px;\n                border-bottom: 2px solid %s;\n                padding-left: 3px;\n                padding-right: 3px;\n                font-size: %s;\n            }' % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.hStyle)

    def setDim(self, d):
        if self.dim != d:
            self.dim = d
            self.updateStyle()

    def setOrientation(self, o):
        VerticalLabel.setOrientation(self, o)
        self.updateStyle()

    def isClosable(self):
        return self.closeButton is not None

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.pressPos = lpos
        self.mouseMoved = False
        ev.accept()

    def mouseMoveEvent(self, ev):
        if not self.mouseMoved:
            lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
            self.mouseMoved = (lpos - self.pressPos).manhattanLength() > QtWidgets.QApplication.startDragDistance()
        if self.mouseMoved and ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.dock.startDrag()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        ev.accept()
        if not self.mouseMoved:
            self.sigClicked.emit(self, ev)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dock.float()

    def resizeEvent(self, ev):
        if self.closeButton:
            if self.orientation == 'vertical':
                size = ev.size().width()
                pos = QtCore.QPoint(0, 0)
            else:
                size = ev.size().height()
                pos = QtCore.QPoint(ev.size().width() - size, 0)
            self.closeButton.setFixedSize(QtCore.QSize(size, size))
            self.closeButton.move(pos)
        super(DockLabel, self).resizeEvent(ev)