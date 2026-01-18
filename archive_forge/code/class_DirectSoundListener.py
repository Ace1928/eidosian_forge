import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
class DirectSoundListener:

    def __init__(self, native_listener):
        self._native_listener = native_listener

    def delete(self):
        if self._native_listener:
            self._native_listener.Release()
            self._native_listener = None

    @property
    def position(self):
        vector = lib.D3DVECTOR()
        _check(self._native_listener.GetPosition(ctypes.byref(vector)))
        return (vector.x, vector.y, vector.z)

    @position.setter
    def position(self, value):
        _check(self._native_listener.SetPosition(*list(value) + [lib.DS3D_IMMEDIATE]))

    @property
    def orientation(self):
        front = lib.D3DVECTOR()
        top = lib.D3DVECTOR()
        _check(self._native_listener.GetOrientation(ctypes.byref(front), ctypes.byref(top)))
        return (front.x, front.y, front.z, top.x, top.y, top.z)

    @orientation.setter
    def orientation(self, orientation):
        _check(self._native_listener.SetOrientation(*list(orientation) + [lib.DS3D_IMMEDIATE]))