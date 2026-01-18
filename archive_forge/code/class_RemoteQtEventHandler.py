import atexit
import inspect
import multiprocessing.connection
import os
import signal
import subprocess
import sys
import time
import pickle
from ..Qt import QT_LIB, mkQApp
from ..util import cprint  # color printing for debugging
from .remoteproxy import (
import threading
class RemoteQtEventHandler(RemoteEventHandler):

    def __init__(self, *args, **kwds):
        RemoteEventHandler.__init__(self, *args, **kwds)

    def startEventTimer(self):
        from ..Qt import QtCore
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.processRequests)
        self.timer.start(10)

    def processRequests(self):
        try:
            RemoteEventHandler.processRequests(self)
        except ClosedError:
            from ..Qt import QtWidgets
            QtWidgets.QApplication.instance().quit()
            self.timer.stop()