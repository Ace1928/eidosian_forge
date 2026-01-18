import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_timing_info(Structure):
    __slots__ = ['timestamp', 'synchronized_clocks', 'sink_usec', 'source_usec', 'transport_usec', 'playing', 'write_index_corrupt', 'write_index', 'read_index_corrupt', 'read_index', 'configured_sink_usec', 'configured_source_usec', 'since_underrun']