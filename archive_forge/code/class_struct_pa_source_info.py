import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_source_info(Structure):
    __slots__ = ['name', 'index', 'description', 'sample_spec', 'channel_map', 'owner_module', 'volume', 'mute', 'monitor_of_sink', 'monitor_of_sink_name', 'latency', 'driver', 'flags', 'proplist', 'configured_latency', 'base_volume', 'state', 'n_volume_steps', 'card', 'n_ports', 'ports', 'active_port', 'n_formats', 'formats']