import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_source_output_info(Structure):
    __slots__ = ['index', 'name', 'owner_module', 'client', 'source', 'sample_spec', 'channel_map', 'buffer_usec', 'source_usec', 'resample_method', 'driver', 'proplist', 'corked', 'volume', 'mute', 'has_volume', 'volume_writable', 'format']