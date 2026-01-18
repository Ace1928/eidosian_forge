import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_sink_info(Structure):
    __slots__ = ['name', 'index', 'description', 'sample_spec', 'channel_map', 'owner_module', 'volume', 'mute', 'monitor_source', 'monitor_source_name', 'latency', 'driver', 'flags', 'proplist', 'configured_latency', 'base_volume', 'state', 'n_volume_steps', 'card', 'n_ports', 'ports', 'active_port', 'n_formats', 'formats']