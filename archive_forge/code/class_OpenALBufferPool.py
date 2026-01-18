import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
class OpenALBufferPool(OpenALObject):
    """At least Mac OS X doesn't free buffers when a source is deleted; it just
    detaches them from the source.  So keep our own recycled queue.
    """

    def __init__(self):
        self._buffers = []
        'List of free buffer names'

    def __len__(self):
        return len(self._buffers)

    def delete(self):
        assert _debug('Delete interface.OpenALBufferPool')
        while self._buffers:
            self._buffers.pop().delete()

    def get_buffer(self):
        """Convenience for returning one buffer name"""
        return self.get_buffers(1)[0]

    def get_buffers(self, number):
        """Returns an array containing `number` buffer names.  The returned list must
        not be modified in any way, and may get changed by subsequent calls to
        get_buffers.
        """
        ret_buffers = []
        while self._buffers and len(ret_buffers) < number:
            b = self._buffers.pop()
            if b.is_valid:
                ret_buffers.append(b)
        if (missing := (number - len(ret_buffers))) > 0:
            names = (al.ALuint * missing)()
            al.alGenBuffers(missing, names)
            self._check_error('Error generating buffers.')
            ret_buffers.extend((OpenALBuffer(al.ALuint(name)) for name in names))
        return ret_buffers

    def return_buffers(self, buffers):
        """Throw buffers not used anymore back into the pool."""
        for buf in buffers:
            if buf.is_valid:
                self._buffers.append(buf)