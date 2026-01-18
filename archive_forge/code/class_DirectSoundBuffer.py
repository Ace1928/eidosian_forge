import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
class DirectSoundBuffer:

    def __init__(self, native_buffer, audio_format, buffer_size):
        self.audio_format = audio_format
        self.buffer_size = buffer_size
        self._native_buffer = native_buffer
        if audio_format is not None and audio_format.channels == 1:
            self._native_buffer3d = lib.IDirectSound3DBuffer()
            self._native_buffer.QueryInterface(lib.IID_IDirectSound3DBuffer, ctypes.byref(self._native_buffer3d))
        else:
            self._native_buffer3d = None

    def delete(self):
        if self._native_buffer is not None:
            self._native_buffer.Stop()
            self._native_buffer.Release()
            self._native_buffer = None
            if self._native_buffer3d is not None:
                self._native_buffer3d.Release()
                self._native_buffer3d = None

    @property
    def volume(self):
        vol = lib.LONG()
        _check(self._native_buffer.GetVolume(ctypes.byref(vol)))
        return vol.value

    @volume.setter
    def volume(self, value):
        _check(self._native_buffer.SetVolume(value))

    @property
    def current_position(self):
        """Tuple of current play position and current write position.
        Only play position can be modified, so setter only accepts a single value."""
        play_cursor = lib.DWORD()
        write_cursor = lib.DWORD()
        _check(self._native_buffer.GetCurrentPosition(play_cursor, write_cursor))
        return _CurrentPosition(play_cursor.value, write_cursor.value)

    @current_position.setter
    def current_position(self, value):
        _check(self._native_buffer.SetCurrentPosition(value))

    @property
    def is3d(self):
        return self._native_buffer3d is not None

    @property
    def is_playing(self):
        return self._get_status() & lib.DSBSTATUS_PLAYING != 0

    @property
    def is_buffer_lost(self):
        return self._get_status() & lib.DSBSTATUS_BUFFERLOST != 0

    def _get_status(self):
        status = lib.DWORD()
        _check(self._native_buffer.GetStatus(status))
        return status.value

    @property
    def position(self):
        if self.is3d:
            position = lib.D3DVECTOR()
            _check(self._native_buffer3d.GetPosition(ctypes.byref(position)))
            return (position.x, position.y, position.z)
        else:
            return (0, 0, 0)

    @position.setter
    def position(self, position):
        if self.is3d:
            x, y, z = position
            _check(self._native_buffer3d.SetPosition(x, y, z, lib.DS3D_IMMEDIATE))

    @property
    def min_distance(self):
        """The minimum distance, which is the distance from the
        listener at which sounds in this buffer begin to be attenuated."""
        if self.is3d:
            value = lib.D3DVALUE()
            _check(self._native_buffer3d.GetMinDistance(ctypes.byref(value)))
            return value.value
        else:
            return 0

    @min_distance.setter
    def min_distance(self, value):
        if self.is3d:
            _check(self._native_buffer3d.SetMinDistance(value, lib.DS3D_IMMEDIATE))

    @property
    def max_distance(self):
        """The maximum distance, which is the distance from the listener beyond which
        sounds in this buffer are no longer attenuated."""
        if self.is3d:
            value = lib.D3DVALUE()
            _check(self._native_buffer3d.GetMaxDistance(ctypes.byref(value)))
            return value.value
        else:
            return 0

    @max_distance.setter
    def max_distance(self, value):
        if self.is3d:
            _check(self._native_buffer3d.SetMaxDistance(value, lib.DS3D_IMMEDIATE))

    @property
    def frequency(self):
        value = lib.DWORD()
        _check(self._native_buffer.GetFrequency(value))
        return value.value

    @frequency.setter
    def frequency(self, value):
        """The frequency, in samples per second, at which the buffer is playing."""
        _check(self._native_buffer.SetFrequency(value))

    @property
    def cone_orientation(self):
        """The orientation of the sound projection cone."""
        if self.is3d:
            orientation = lib.D3DVECTOR()
            _check(self._native_buffer3d.GetConeOrientation(ctypes.byref(orientation)))
            return (orientation.x, orientation.y, orientation.z)
        else:
            return (0, 0, 0)

    @cone_orientation.setter
    def cone_orientation(self, value):
        if self.is3d:
            x, y, z = value
            _check(self._native_buffer3d.SetConeOrientation(x, y, z, lib.DS3D_IMMEDIATE))
    _ConeAngles = namedtuple('_ConeAngles', ['inside', 'outside'])

    @property
    def cone_angles(self):
        """The inside and outside angles of the sound projection cone."""
        if self.is3d:
            inside = lib.DWORD()
            outside = lib.DWORD()
            _check(self._native_buffer3d.GetConeAngles(ctypes.byref(inside), ctypes.byref(outside)))
            return self._ConeAngles(inside.value, outside.value)
        else:
            return self._ConeAngles(0, 0)

    def set_cone_angles(self, inside, outside):
        """The inside and outside angles of the sound projection cone."""
        if self.is3d:
            _check(self._native_buffer3d.SetConeAngles(inside, outside, lib.DS3D_IMMEDIATE))

    @property
    def cone_outside_volume(self):
        """The volume of the sound outside the outside angle of the sound projection cone."""
        if self.is3d:
            volume = lib.LONG()
            _check(self._native_buffer3d.GetConeOutsideVolume(ctypes.byref(volume)))
            return volume.value
        else:
            return 0

    @cone_outside_volume.setter
    def cone_outside_volume(self, value):
        if self.is3d:
            _check(self._native_buffer3d.SetConeOutsideVolume(value, lib.DS3D_IMMEDIATE))

    def create_listener(self):
        native_listener = lib.IDirectSound3DListener()
        self._native_buffer.QueryInterface(lib.IID_IDirectSound3DListener, ctypes.byref(native_listener))
        return DirectSoundListener(native_listener)

    def play(self):
        _check(self._native_buffer.Play(0, 0, lib.DSBPLAY_LOOPING))

    def stop(self):
        _check(self._native_buffer.Stop())

    class _WritePointer:

        def __init__(self):
            self.audio_ptr_1 = ctypes.c_void_p()
            self.audio_length_1 = lib.DWORD()
            self.audio_ptr_2 = ctypes.c_void_p()
            self.audio_length_2 = lib.DWORD()

    def lock(self, write_cursor, write_size):
        assert _debug('DirectSoundBuffer.lock({}, {})'.format(write_cursor, write_size))
        pointer = self._WritePointer()
        _check(self._native_buffer.Lock(write_cursor, write_size, ctypes.byref(pointer.audio_ptr_1), ctypes.byref(pointer.audio_length_1), ctypes.byref(pointer.audio_ptr_2), ctypes.byref(pointer.audio_length_2), 0))
        return pointer

    def unlock(self, pointer):
        _check(self._native_buffer.Unlock(pointer.audio_ptr_1, pointer.audio_length_1, pointer.audio_ptr_2, pointer.audio_length_2))