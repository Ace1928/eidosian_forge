import array
import ctypes.wintypes
import platform
import struct
from paramiko.common import zero_byte
from paramiko.util import b
import _thread as thread
from . import _winapi
def _get_pageant_window_object():
    return ctypes.windll.user32.FindWindowA(b'Pageant', b'Pageant')