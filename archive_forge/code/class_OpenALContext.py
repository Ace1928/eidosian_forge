import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
class OpenALContext(OpenALObject):

    def __init__(self, device, al_context):
        self.device = device
        self._al_context = al_context
        self._sources = set()
        self.make_current()

    def delete_sources(self):
        for s in tuple(self._sources):
            s.delete()

    def delete(self):
        assert _debug('Delete interface.OpenALContext')
        if ctypes.cast(alc.alcGetCurrentContext(), ctypes.c_void_p).value == ctypes.cast(self._al_context, ctypes.c_void_p).value:
            alc.alcMakeContextCurrent(None)
            self.device.check_context_error('Failed to make context no longer current.')
        alc.alcDestroyContext(self._al_context)
        self.device.check_context_error('Failed to destroy context.')
        self._al_context = None

    def make_current(self):
        alc.alcMakeContextCurrent(self._al_context)
        self.device.check_context_error('Failed to make context current.')

    def create_source(self):
        self.make_current()
        new_source = OpenALSource(self)
        self._sources.add(new_source)
        return new_source

    def source_deleted(self, source):
        self._sources.remove(source)