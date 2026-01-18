import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
class _BaseWaitHandleFuture(futures.Future):
    """Subclass of Future which represents a wait handle."""

    def __init__(self, ov, handle, wait_handle, *, loop=None):
        super().__init__(loop=loop)
        if self._source_traceback:
            del self._source_traceback[-1]
        self._ov = ov
        self._handle = handle
        self._wait_handle = wait_handle
        self._registered = True

    def _poll(self):
        return _winapi.WaitForSingleObject(self._handle, 0) == _winapi.WAIT_OBJECT_0

    def _repr_info(self):
        info = super()._repr_info()
        info.append(f'handle={self._handle:#x}')
        if self._handle is not None:
            state = 'signaled' if self._poll() else 'waiting'
            info.append(state)
        if self._wait_handle is not None:
            info.append(f'wait_handle={self._wait_handle:#x}')
        return info

    def _unregister_wait_cb(self, fut):
        self._ov = None

    def _unregister_wait(self):
        if not self._registered:
            return
        self._registered = False
        wait_handle = self._wait_handle
        self._wait_handle = None
        try:
            _overlapped.UnregisterWait(wait_handle)
        except OSError as exc:
            if exc.winerror != _overlapped.ERROR_IO_PENDING:
                context = {'message': 'Failed to unregister the wait handle', 'exception': exc, 'future': self}
                if self._source_traceback:
                    context['source_traceback'] = self._source_traceback
                self._loop.call_exception_handler(context)
                return
        self._unregister_wait_cb(None)

    def cancel(self, msg=None):
        self._unregister_wait()
        return super().cancel(msg=msg)

    def set_exception(self, exception):
        self._unregister_wait()
        super().set_exception(exception)

    def set_result(self, result):
        self._unregister_wait()
        super().set_result(result)