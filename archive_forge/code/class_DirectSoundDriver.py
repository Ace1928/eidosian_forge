import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
class DirectSoundDriver:

    def __init__(self):
        assert _debug('Constructing DirectSoundDriver')
        self._native_dsound = lib.IDirectSound()
        _check(lib.DirectSoundCreate(None, ctypes.byref(self._native_dsound), None))
        hwnd = _user32.GetDesktopWindow()
        _check(self._native_dsound.SetCooperativeLevel(hwnd, lib.DSSCL_NORMAL))
        self.primary_buffer = self._create_primary_buffer()

    def delete(self):
        self.primary_buffer.delete()
        self.primary_buffer = None
        self._native_dsound.Release()

    def create_buffer(self, audio_format, buffer_size):
        wave_format = _create_wave_format(audio_format)
        buffer_desc = _create_buffer_desc(wave_format, buffer_size)
        return DirectSoundBuffer(self._create_native_buffer(buffer_desc), audio_format, buffer_size)

    def create_listener(self):
        return self.primary_buffer.create_listener()

    def _create_primary_buffer(self):
        return DirectSoundBuffer(self._create_native_buffer(_create_primary_buffer_desc()), None, 0)

    def _create_native_buffer(self, buffer_desc):
        buf = lib.IDirectSoundBuffer()
        _check(self._native_dsound.CreateSoundBuffer(buffer_desc, ctypes.byref(buf), None))
        return buf