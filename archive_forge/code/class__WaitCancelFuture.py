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
class _WaitCancelFuture(_BaseWaitHandleFuture):
    """Subclass of Future which represents a wait for the cancellation of a
    _WaitHandleFuture using an event.
    """

    def __init__(self, ov, event, wait_handle, *, loop=None):
        super().__init__(ov, event, wait_handle, loop=loop)
        self._done_callback = None

    def cancel(self):
        raise RuntimeError('_WaitCancelFuture must not be cancelled')

    def set_result(self, result):
        super().set_result(result)
        if self._done_callback is not None:
            self._done_callback(self)

    def set_exception(self, exception):
        super().set_exception(exception)
        if self._done_callback is not None:
            self._done_callback(self)