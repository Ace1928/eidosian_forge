from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
class DictEntry(Struct):

    def __init__(self, fields):
        if len(fields) != 2:
            raise TypeError('Dict entry must have 2 fields, not %d' % len(fields))
        if not isinstance(fields[0], (FixedType, StringType)):
            raise TypeError('First field in dict entry must be simple type, not {}'.format(type(fields[0])))
        super().__init__(fields)