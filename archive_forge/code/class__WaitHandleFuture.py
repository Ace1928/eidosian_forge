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
class _WaitHandleFuture(_BaseWaitHandleFuture):

    def __init__(self, ov, handle, wait_handle, proactor, *, loop=None):
        super().__init__(ov, handle, wait_handle, loop=loop)
        self._proactor = proactor
        self._unregister_proactor = True
        self._event = _overlapped.CreateEvent(None, True, False, None)
        self._event_fut = None

    def _unregister_wait_cb(self, fut):
        if self._event is not None:
            _winapi.CloseHandle(self._event)
            self._event = None
            self._event_fut = None
        self._proactor._unregister(self._ov)
        self._proactor = None
        super()._unregister_wait_cb(fut)

    def _unregister_wait(self):
        if not self._registered:
            return
        self._registered = False
        wait_handle = self._wait_handle
        self._wait_handle = None
        try:
            _overlapped.UnregisterWaitEx(wait_handle, self._event)
        except OSError as exc:
            if exc.winerror != _overlapped.ERROR_IO_PENDING:
                context = {'message': 'Failed to unregister the wait handle', 'exception': exc, 'future': self}
                if self._source_traceback:
                    context['source_traceback'] = self._source_traceback
                self._loop.call_exception_handler(context)
                return
        self._event_fut = self._proactor._wait_cancel(self._event, self._unregister_wait_cb)