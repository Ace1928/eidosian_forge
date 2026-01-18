import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
class OpenALSource(OpenALObject):

    def __init__(self, context):
        self.context = context
        self._al_source = al.ALuint()
        al.alGenSources(1, self._al_source)
        self._check_error('Failed to create source.')
        self.buffer_pool = context.device.buffer_pool
        self._state = None
        self._get_state()
        self._owned_buffers = {}

    def delete(self):
        if self.context is None:
            assert _debug('Delete interface.OpenAlSource on deleted source, ignoring')
            return
        assert _debug('Delete interface.OpenALSource')
        al.alDeleteSources(1, self._al_source)
        self._check_error('Failed to delete source.')
        self.context.source_deleted(self)
        self.buffer_pool = None
        self.context = None
        self._al_source = None

    @property
    def is_initial(self):
        self._get_state()
        return self._state == al.AL_INITIAL

    @property
    def is_playing(self):
        self._get_state()
        return self._state == al.AL_PLAYING

    @property
    def is_paused(self):
        self._get_state()
        return self._state == al.AL_PAUSED

    @property
    def is_stopped(self):
        self._get_state()
        return self._state == al.AL_STOPPED

    def _int_source_property(attribute):
        return property(lambda self: self._get_int(attribute), lambda self, value: self._set_int(attribute, value))

    def _float_source_property(attribute):
        return property(lambda self: self._get_float(attribute), lambda self, value: self._set_float(attribute, value))

    def _3floats_source_property(attribute):
        return property(lambda self: self._get_3floats(attribute), lambda self, value: self._set_3floats(attribute, value))
    position = _3floats_source_property(al.AL_POSITION)
    velocity = _3floats_source_property(al.AL_VELOCITY)
    gain = _float_source_property(al.AL_GAIN)
    buffers_queued = _int_source_property(al.AL_BUFFERS_QUEUED)
    buffers_processed = _int_source_property(al.AL_BUFFERS_PROCESSED)
    min_gain = _float_source_property(al.AL_MIN_GAIN)
    max_gain = _float_source_property(al.AL_MAX_GAIN)
    reference_distance = _float_source_property(al.AL_REFERENCE_DISTANCE)
    rolloff_factor = _float_source_property(al.AL_ROLLOFF_FACTOR)
    pitch = _float_source_property(al.AL_PITCH)
    max_distance = _float_source_property(al.AL_MAX_DISTANCE)
    direction = _3floats_source_property(al.AL_DIRECTION)
    cone_inner_angle = _float_source_property(al.AL_CONE_INNER_ANGLE)
    cone_outer_angle = _float_source_property(al.AL_CONE_OUTER_ANGLE)
    cone_outer_gain = _float_source_property(al.AL_CONE_OUTER_GAIN)
    sec_offset = _float_source_property(al.AL_SEC_OFFSET)
    sample_offset = _int_source_property(al.AL_SAMPLE_OFFSET)
    byte_offset = _int_source_property(al.AL_BYTE_OFFSET)
    del _int_source_property
    del _float_source_property
    del _3floats_source_property

    def play(self):
        al.alSourcePlay(self._al_source)
        self._check_error('Failed to play source.')

    def pause(self):
        al.alSourcePause(self._al_source)
        self._check_error('Failed to pause source.')

    def stop(self):
        al.alSourceStop(self._al_source)
        self._check_error('Failed to stop source.')

    def clear(self):
        self._set_int(al.AL_BUFFER, al.AL_NONE)
        self.buffer_pool.return_buffers(self._owned_buffers.values())
        self._owned_buffers.clear()

    def get_buffer(self):
        return self.buffer_pool.get_buffer()

    def queue_buffer(self, buf):
        assert buf.is_valid
        al.alSourceQueueBuffers(self._al_source, 1, ctypes.byref(buf.al_name))
        self._check_error('Failed to queue buffer.')
        self._owned_buffers[buf.name] = buf

    def unqueue_buffers(self):
        processed = self.buffers_processed
        assert _debug('Processed buffer count: {}'.format(processed))
        if processed > 0:
            buffers = (al.ALuint * processed)()
            al.alSourceUnqueueBuffers(self._al_source, len(buffers), buffers)
            self._check_error('Failed to unqueue buffers from source.')
            self.buffer_pool.return_buffers([self._owned_buffers.pop(bn) for bn in buffers])
        return processed

    def _get_state(self):
        if self._al_source is not None:
            self._state = self._get_int(al.AL_SOURCE_STATE)

    def _get_int(self, key):
        assert self._al_source is not None
        al_int = al.ALint()
        al.alGetSourcei(self._al_source, key, al_int)
        self._check_error('Failed to get value')
        return al_int.value

    def _set_int(self, key, value):
        assert self._al_source is not None
        al.alSourcei(self._al_source, key, int(value))
        self._check_error('Failed to set value.')

    def _get_float(self, key):
        assert self._al_source is not None
        al_float = al.ALfloat()
        al.alGetSourcef(self._al_source, key, al_float)
        self._check_error('Failed to get value')
        return al_float.value

    def _set_float(self, key, value):
        assert self._al_source is not None
        al.alSourcef(self._al_source, key, float(value))
        self._check_error('Failed to set value.')

    def _get_3floats(self, key):
        assert self._al_source is not None
        x = al.ALfloat()
        y = al.ALfloat()
        z = al.ALfloat()
        al.alGetSource3f(self._al_source, key, x, y, z)
        self._check_error('Failed to get value')
        return (x.value, y.value, z.value)

    def _set_3floats(self, key, values):
        assert self._al_source is not None
        x, y, z = map(float, values)
        al.alSource3f(self._al_source, key, x, y, z)
        self._check_error('Failed to set value.')