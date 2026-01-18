import xcffib
import struct
import io
class ModMask:
    Shift = 1 << 0
    Lock = 1 << 1
    Control = 1 << 2
    _1 = 1 << 3
    _2 = 1 << 4
    _3 = 1 << 5
    _4 = 1 << 6
    _5 = 1 << 7
    Any = 1 << 15