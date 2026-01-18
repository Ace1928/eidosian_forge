from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
def _libc_clock_gettime_wrapper():
    from os import strerror

    class struct_tv(ctypes.Structure):
        _fields_ = [('tv_sec', ctypes.c_long), ('tv_usec', ctypes.c_long)]
    _clock_gettime = _libc.clock_gettime
    _clock_gettime.argtypes = [ctypes.c_long, ctypes.POINTER(struct_tv)]
    if 'linux' in platform:
        _clockid = 4
    elif 'freebsd' in platform:
        _clockid = 11
        Logger.debug('clock.py: {{{:s}}} clock ID {:d}'.format(platform, _clockid))
    elif 'openbsd' in platform:
        _clockid = 3
    else:
        _clockid = 1
    tv = struct_tv()

    def _time():
        if _clock_gettime(ctypes.c_long(_clockid), ctypes.pointer(tv)) != 0:
            _ernno = ctypes.get_errno()
            raise OSError(_ernno, strerror(_ernno))
        return tv.tv_sec + tv.tv_usec * 1e-09
    return _time