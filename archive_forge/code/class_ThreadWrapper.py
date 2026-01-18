from __future__ import nested_scopes
from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log
class ThreadWrapper(QtCore.QThread):

    def __init__(self, *args, **kwargs):
        _original_thread_init(self, *args, **kwargs)
        if self.__class__.run == _original_QThread.run:
            self.run = self._exec_run
        else:
            self._original_run = self.run
            self.run = self._new_run
        self._original_started = self.started
        self.started = StartedSignalWrapper(self, self.started)

    def _exec_run(self):
        set_trace_in_qt()
        self.exec_()
        return None

    def _new_run(self):
        set_trace_in_qt()
        return self._original_run()