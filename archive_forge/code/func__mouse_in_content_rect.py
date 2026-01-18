from ctypes import *
import pyglet
from pyglet.window import BaseWindow
from pyglet.window import MouseCursor, DefaultMouseCursor
from pyglet.window import WindowException
from pyglet.event import EventDispatcher
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, CGPoint, AutoReleasePool
from .systemcursor import SystemCursor
from .pyglet_delegate import PygletDelegate
from .pyglet_window import PygletWindow, PygletToolWindow
from .pyglet_view import PygletView
def _mouse_in_content_rect(self):
    point = NSEvent.mouseLocation()
    window_frame = self._nswindow.frame()
    rect = self._nswindow.contentRectForFrameRect_(window_frame)
    return cocoapy.foundation.NSMouseInRect(point, rect, False)