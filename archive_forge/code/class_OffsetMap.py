from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class OffsetMap(object):

    def __init__(self, core):
        self.offsets = [(0, 0, core)]

    def add(self, offset, opcode, things):
        self.offsets.append((offset, opcode, things))
        self.offsets.sort(key=lambda x: x[0], reverse=True)

    def get_extension_item(self, extension, item):
        try:
            _, _, things = next(((k, opcode, v) for k, opcode, v in self.offsets if opcode == extension))
            return things[item]
        except StopIteration:
            raise IndexError(item)

    def __getitem__(self, item):
        try:
            offset, _, things = next(((k, opcode, v) for k, opcode, v in self.offsets if item >= k))
            return things[item - offset]
        except StopIteration:
            raise IndexError(item)