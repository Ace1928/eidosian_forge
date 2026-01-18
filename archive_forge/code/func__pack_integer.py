import json
import mmap
import os
import struct
from typing import List
def _pack_integer(data, pos, value):
    data[pos:pos + 4] = _pack_integer_func(value)