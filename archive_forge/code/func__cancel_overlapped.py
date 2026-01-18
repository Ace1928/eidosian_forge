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
def _cancel_overlapped(self):
    if self._ov is None:
        return
    try:
        self._ov.cancel()
    except OSError as exc:
        context = {'message': 'Cancelling an overlapped future failed', 'exception': exc, 'future': self}
        if self._source_traceback:
            context['source_traceback'] = self._source_traceback
        self._loop.call_exception_handler(context)
    self._ov = None