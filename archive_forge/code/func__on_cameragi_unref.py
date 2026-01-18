from gi.repository import Gst
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.camera import CameraBase
from kivy.support import install_gobject_iteration
from kivy.logger import Logger
from ctypes import Structure, c_void_p, c_int, string_at
from weakref import ref
import atexit
def _on_cameragi_unref(obj):
    if obj in CameraGi._instances:
        CameraGi._instances.remove(obj)