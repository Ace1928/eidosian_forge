from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
class OTTableReader(object):
    """Helper class to retrieve data from an OpenType table."""
    __slots__ = ('data', 'offset', 'pos', 'localState', 'tableTag')

    def __init__(self, data, localState=None, offset=0, tableTag=None):
        self.data = data
        self.offset = offset
        self.pos = offset
        self.localState = localState
        self.tableTag = tableTag

    def advance(self, count):
        self.pos += count

    def seek(self, pos):
        self.pos = pos

    def copy(self):
        other = self.__class__(self.data, self.localState, self.offset, self.tableTag)
        other.pos = self.pos
        return other

    def getSubReader(self, offset):
        offset = self.offset + offset
        return self.__class__(self.data, self.localState, offset, self.tableTag)

    def readValue(self, typecode, staticSize):
        pos = self.pos
        newpos = pos + staticSize
        value, = struct.unpack(f'>{typecode}', self.data[pos:newpos])
        self.pos = newpos
        return value

    def readArray(self, typecode, staticSize, count):
        pos = self.pos
        newpos = pos + count * staticSize
        value = array.array(typecode, self.data[pos:newpos])
        if sys.byteorder != 'big':
            value.byteswap()
        self.pos = newpos
        return value.tolist()

    def readInt8(self):
        return self.readValue('b', staticSize=1)

    def readInt8Array(self, count):
        return self.readArray('b', staticSize=1, count=count)

    def readShort(self):
        return self.readValue('h', staticSize=2)

    def readShortArray(self, count):
        return self.readArray('h', staticSize=2, count=count)

    def readLong(self):
        return self.readValue('i', staticSize=4)

    def readLongArray(self, count):
        return self.readArray('i', staticSize=4, count=count)

    def readUInt8(self):
        return self.readValue('B', staticSize=1)

    def readUInt8Array(self, count):
        return self.readArray('B', staticSize=1, count=count)

    def readUShort(self):
        return self.readValue('H', staticSize=2)

    def readUShortArray(self, count):
        return self.readArray('H', staticSize=2, count=count)

    def readULong(self):
        return self.readValue('I', staticSize=4)

    def readULongArray(self, count):
        return self.readArray('I', staticSize=4, count=count)

    def readUInt24(self):
        pos = self.pos
        newpos = pos + 3
        value, = struct.unpack('>l', b'\x00' + self.data[pos:newpos])
        self.pos = newpos
        return value

    def readUInt24Array(self, count):
        return [self.readUInt24() for _ in range(count)]

    def readTag(self):
        pos = self.pos
        newpos = pos + 4
        value = Tag(self.data[pos:newpos])
        assert len(value) == 4, value
        self.pos = newpos
        return value

    def readData(self, count):
        pos = self.pos
        newpos = pos + count
        value = self.data[pos:newpos]
        self.pos = newpos
        return value

    def __setitem__(self, name, value):
        state = self.localState.copy() if self.localState else dict()
        state[name] = value
        self.localState = state

    def __getitem__(self, name):
        return self.localState and self.localState[name]

    def __contains__(self, name):
        return self.localState and name in self.localState