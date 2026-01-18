import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
class WintabTablet(Tablet):

    def __init__(self, index):
        self._device = wintab.WTI_DEVICES + index
        self.name = wtinfo_string(self._device, wintab.DVC_NAME).strip()
        self.id = wtinfo_string(self._device, wintab.DVC_PNPID)
        hardware = wtinfo_uint(self._device, wintab.DVC_HARDWARE)
        n_cursors = wtinfo_uint(self._device, wintab.DVC_NCSRTYPES)
        first_cursor = wtinfo_uint(self._device, wintab.DVC_FIRSTCSR)
        self.pressure_axis = wtinfo(self._device, wintab.DVC_NPRESSURE, wintab.AXIS())
        self.cursors = []
        self._cursor_map = {}
        for i in range(n_cursors):
            cursor = WintabTabletCursor(self, i + first_cursor)
            if not cursor.bogus:
                self.cursors.append(cursor)
                self._cursor_map[i + first_cursor] = cursor

    def open(self, window):
        return WintabTabletCanvas(self, window)