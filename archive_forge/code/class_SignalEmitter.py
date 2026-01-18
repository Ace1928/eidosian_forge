import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
class SignalEmitter(pg.QtCore.QObject):
    signal = pg.QtCore.Signal(object, object)

    def __init__(self, queued):
        pg.QtCore.QObject.__init__(self)
        if queued:
            self.signal.connect(self.run, pg.QtCore.Qt.ConnectionType.QueuedConnection)
        else:
            self.signal.connect(self.run)

    def run(self, func, args):
        func(*args)