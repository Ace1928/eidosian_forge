import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_stat_info(Structure):
    __slots__ = ['memblock_total', 'memblock_total_size', 'memblock_allocated', 'memblock_allocated_size', 'scache_size']