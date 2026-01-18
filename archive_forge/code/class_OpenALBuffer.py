import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
class OpenALBuffer(OpenALObject):
    _format_map = {(1, 8): al.AL_FORMAT_MONO8, (1, 16): al.AL_FORMAT_MONO16, (2, 8): al.AL_FORMAT_STEREO8, (2, 16): al.AL_FORMAT_STEREO16}

    def __init__(self, al_name):
        self.al_name = al_name
        self.name = al_name.value
        assert self.is_valid

    @property
    def is_valid(self):
        self._check_error('Before validate buffer.')
        if self.al_name is None:
            return False
        valid = bool(al.alIsBuffer(self.al_name))
        if not valid:
            al.alGetError()
        return valid

    def delete(self):
        if self.al_name is not None and self.is_valid:
            al.alDeleteBuffers(1, ctypes.byref(self.al_name))
            self._check_error('Error deleting buffer.')
            self.al_name = None

    def data(self, audio_data, audio_format):
        assert self.is_valid
        try:
            al_format = self._format_map[audio_format.channels, audio_format.sample_size]
        except KeyError:
            raise MediaException(f"OpenAL does not support '{audio_format.sample_size}bit' audio.")
        al.alBufferData(self.al_name, al_format, audio_data.pointer, audio_data.length, audio_format.sample_rate)
        self._check_error('Failed to add data to buffer.')