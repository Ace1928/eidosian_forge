from pyglet.input.base import Tablet, TabletCanvas
from pyglet.input.base import TabletCursor, DeviceOpenException
from pyglet.input.linux.x11_xinput import XInputWindowEventDispatcher
from pyglet.input.linux.x11_xinput import get_devices, DeviceResponder
def _motion(self, e):
    cursor = self._cursor_map.get(e.deviceid)
    x = e.x
    y = self.window.height - e.y
    pressure = e.axis_data[2] / float(cursor.max_pressure)
    self.dispatch_event('on_motion', cursor, x, y, pressure, 0.0, 0.0, 0.0)