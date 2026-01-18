import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
@pyglet.window.win32.Win32EventHandler(0)
def _event_wt_packet(self, msg, wParam, lParam):
    if lParam != self._context:
        return
    packet = wintab.PACKET()
    if lib.WTPacket(self._context, wParam, ctypes.byref(packet)) == 0:
        return
    if not packet.pkChanged:
        return
    window_x, window_y = self.window.get_location()
    window_y = self.window.screen.height - window_y - self.window.height
    x = packet.pkX - window_x
    y = packet.pkY - window_y
    pressure = (packet.pkNormalPressure + self._pressure_bias) * self._pressure_scale
    if self._current_cursor is None:
        self._set_current_cursor(packet.pkCursor)
    self.dispatch_event('on_motion', self._current_cursor, x, y, pressure, 0.0, 0.0, packet.pkButtons)