from __future__ import nested_scopes
from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log
class RunnableWrapper(QtCore.QRunnable):

    def __init__(self, *args, **kwargs):
        _original_runnable_init(self, *args, **kwargs)
        self._original_run = self.run
        self.run = self._new_run

    def _new_run(self):
        set_trace_in_qt()
        return self._original_run()