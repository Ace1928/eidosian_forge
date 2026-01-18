import struct
from llvmlite.ir._utils import _StrCaching
class _Repeat(object):

    def __init__(self, value, size):
        self.value = value
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if 0 <= item < self.size:
            return self.value
        else:
            raise IndexError(item)