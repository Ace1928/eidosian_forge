from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
import atexit
import enum
import mmap
import os
import sys
import tempfile
from .. import Qt
from .. import CONFIG_OPTIONS
from .. import multiprocess as mp
from .GraphicsView import GraphicsView
class RemoteGraphicsView(QtWidgets.QWidget):
    """
    Replacement for GraphicsView that does all scene management and rendering on a remote process,
    while displaying on the local widget.
    
    GraphicsItems must be created by proxy to the remote process.
    
    """

    def __init__(self, parent=None, *args, **kwds):
        """
        The keyword arguments 'useOpenGL' and 'backgound', if specified, are passed to the remote
        GraphicsView.__init__(). All other keyword arguments are passed to multiprocess.QtProcess.__init__().
        """
        self._img = None
        self._imgReq = None
        self._sizeHint = (640, 480)
        QtWidgets.QWidget.__init__(self)
        remoteKwds = {}
        for kwd in ['useOpenGL', 'background']:
            if kwd in kwds:
                remoteKwds[kwd] = kwds.pop(kwd)
        self._proc = mp.QtProcess(**kwds)
        self.pg = self._proc._import('pyqtgraph')
        self.pg.setConfigOptions(**CONFIG_OPTIONS)
        rpgRemote = self._proc._import('pyqtgraph.widgets.RemoteGraphicsView')
        self._view = rpgRemote.Renderer(*args, **remoteKwds)
        self._view._setProxyOptions(deferGetattr=True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.shm = None
        shmFileName = self._view.shmFileName()
        if sys.platform == 'win32':
            opener = lambda path, flags: os.open(path, flags | os.O_TEMPORARY)
        else:
            opener = None
        self.shmFile = open(shmFileName, 'rb', opener=opener)
        self._view.sceneRendered.connect(mp.proxy(self.remoteSceneChanged))
        for method in ['scene', 'setCentralItem']:
            setattr(self, method, getattr(self._view, method))

    def resizeEvent(self, ev):
        ret = super().resizeEvent(ev)
        self._view.resize(self.size(), _callSync='off')
        return ret

    def sizeHint(self):
        return QtCore.QSize(*self._sizeHint)

    def remoteSceneChanged(self, data):
        w, h, size = data
        if self.shm is None or self.shm.size != size:
            if self.shm is not None:
                self.shm.close()
            self.shm = mmap.mmap(self.shmFile.fileno(), size, access=mmap.ACCESS_READ)
        self._img = QtGui.QImage(self.shm, w, h, QtGui.QImage.Format.Format_RGB32).copy()
        self.update()

    def paintEvent(self, ev):
        if self._img is None:
            return
        p = QtGui.QPainter(self)
        p.drawImage(self.rect(), self._img, self._img.rect())
        p.end()

    def mousePressEvent(self, ev):
        self._view.mousePressEvent(MouseEvent(ev), _callSync='off')
        ev.accept()
        return super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._view.mouseReleaseEvent(MouseEvent(ev), _callSync='off')
        ev.accept()
        return super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        self._view.mouseMoveEvent(MouseEvent(ev), _callSync='off')
        ev.accept()
        return super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        self._view.wheelEvent(WheelEvent(ev), _callSync='off')
        ev.accept()
        return super().wheelEvent(ev)

    def enterEvent(self, ev):
        self._view.enterEvent(EnterEvent(ev), _callSync='off')
        return super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._view.leaveEvent(LeaveEvent(ev), _callSync='off')
        return super().leaveEvent(ev)

    def remoteProcess(self):
        """Return the remote process handle. (see multiprocess.remoteproxy.RemoteEventHandler)"""
        return self._proc

    def close(self):
        """Close the remote process. After this call, the widget will no longer be updated."""
        self._view.sceneRendered.disconnect()
        self._proc.close()