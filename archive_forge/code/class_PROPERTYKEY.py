from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
class PROPERTYKEY(ctypes.Structure):
    _fields_ = [('fmtid', com.GUID), ('pid', DWORD)]

    def __repr__(self):
        return 'PROPERTYKEY({}, pid={})'.format(self.fmtid, self.pid)