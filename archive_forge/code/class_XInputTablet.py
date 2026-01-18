from pyglet.input.base import Tablet, TabletCanvas
from pyglet.input.base import TabletCursor, DeviceOpenException
from pyglet.input.linux.x11_xinput import XInputWindowEventDispatcher
from pyglet.input.linux.x11_xinput import get_devices, DeviceResponder
class XInputTablet(Tablet):
    name = 'XInput Tablet'

    def __init__(self, cursors):
        self.cursors = cursors

    def open(self, window):
        return XInputTabletCanvas(window, self.cursors)