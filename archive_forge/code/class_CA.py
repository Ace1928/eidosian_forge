import xcffib
import struct
import io
from . import xproto
class CA:
    Counter = 1 << 0
    ValueType = 1 << 1
    Value = 1 << 2
    TestType = 1 << 3
    Delta = 1 << 4
    Events = 1 << 5