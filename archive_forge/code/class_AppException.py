import platform
import sys
import weakref
import pyglet
from pyglet import compat_platform
from pyglet.app.base import EventLoop
class AppException(Exception):
    pass