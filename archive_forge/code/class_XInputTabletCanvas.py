from pyglet.input.base import Tablet, TabletCanvas
from pyglet.input.base import TabletCursor, DeviceOpenException
from pyglet.input.linux.x11_xinput import XInputWindowEventDispatcher
from pyglet.input.linux.x11_xinput import get_devices, DeviceResponder
class XInputTabletCanvas(DeviceResponder, TabletCanvas):

    def __init__(self, window, cursors):
        super(XInputTabletCanvas, self).__init__(window)
        self.cursors = cursors
        dispatcher = XInputWindowEventDispatcher.get_dispatcher(window)
        self.display = window.display
        self._open_devices = []
        self._cursor_map = {}
        for cursor in cursors:
            device = cursor.device
            device_id = device._device_id
            self._cursor_map[device_id] = cursor
            cursor.max_pressure = device.axes[2].max
            if self.display._display != device.display._display:
                raise DeviceOpenException('Window and device displays differ')
            open_device = xi.XOpenDevice(device.display._display, device_id)
            if not open_device:
                continue
            self._open_devices.append(open_device)
            dispatcher.open_device(device_id, open_device, self)

    def close(self):
        for device in self._open_devices:
            xi.XCloseDevice(self.display._display, device)

    def _motion(self, e):
        cursor = self._cursor_map.get(e.deviceid)
        x = e.x
        y = self.window.height - e.y
        pressure = e.axis_data[2] / float(cursor.max_pressure)
        self.dispatch_event('on_motion', cursor, x, y, pressure, 0.0, 0.0, 0.0)

    def _proximity_in(self, e):
        cursor = self._cursor_map.get(e.deviceid)
        self.dispatch_event('on_enter', cursor)

    def _proximity_out(self, e):
        cursor = self._cursor_map.get(e.deviceid)
        self.dispatch_event('on_leave', cursor)