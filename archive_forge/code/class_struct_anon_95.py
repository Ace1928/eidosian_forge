import ctypes
from ctypes import *
import pyglet.lib
import pyglet.libs.x11.xlib
class struct_anon_95(Structure):
    __slots__ = ['type', 'serial', 'send_event', 'display', 'window', 'deviceid', 'root', 'subwindow', 'time', 'x', 'y', 'x_root', 'y_root', 'state', 'button', 'same_screen', 'device_state', 'axes_count', 'first_axis', 'axis_data']