import atexit
from ctypes import sizeof
import multiprocessing
import threading
import socket
import time
from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions
class ExceptionAwareProcess(multiprocessing.Process):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exception = None
        self._parent_p, self._child_p = multiprocessing.Pipe()

    def run(self):
        try:
            super().run()
            self._child_p.send(None)
        except Exception as e:
            self._child_p.send(e)

    def join(self):
        super().join()
        if self._parent_p.poll():
            exception = self._parent_p.recv()
            if exception is not None:
                raise exception