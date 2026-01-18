from __future__ import nested_scopes
from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log
def _internal_patch_qt(QtCore, qt_support_mode='auto'):
    pydev_log.debug('Patching Qt: %s', QtCore)
    _original_thread_init = QtCore.QThread.__init__
    _original_runnable_init = QtCore.QRunnable.__init__
    _original_QThread = QtCore.QThread

    class FuncWrapper:

        def __init__(self, original):
            self._original = original

        def __call__(self, *args, **kwargs):
            set_trace_in_qt()
            return self._original(*args, **kwargs)

    class StartedSignalWrapper(QtCore.QObject):
        try:
            _signal = QtCore.Signal()
        except:
            _signal = QtCore.pyqtSignal()

        def __init__(self, thread, original_started):
            QtCore.QObject.__init__(self)
            self.thread = thread
            self.original_started = original_started
            if qt_support_mode in ('pyside', 'pyside2'):
                self._signal = original_started
            else:
                self._signal.connect(self._on_call)
                self.original_started.connect(self._signal)

        def connect(self, func, *args, **kwargs):
            if qt_support_mode in ('pyside', 'pyside2'):
                return self._signal.connect(FuncWrapper(func), *args, **kwargs)
            else:
                return self._signal.connect(func, *args, **kwargs)

        def disconnect(self, *args, **kwargs):
            return self._signal.disconnect(*args, **kwargs)

        def emit(self, *args, **kwargs):
            return self._signal.emit(*args, **kwargs)

        def _on_call(self, *args, **kwargs):
            set_trace_in_qt()

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

    class RunnableWrapper(QtCore.QRunnable):

        def __init__(self, *args, **kwargs):
            _original_runnable_init(self, *args, **kwargs)
            self._original_run = self.run
            self.run = self._new_run

        def _new_run(self):
            set_trace_in_qt()
            return self._original_run()
    QtCore.QThread = ThreadWrapper
    QtCore.QRunnable = RunnableWrapper