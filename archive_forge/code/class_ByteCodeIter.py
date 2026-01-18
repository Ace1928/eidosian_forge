from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
class ByteCodeIter(object):

    def __init__(self, code):
        self.code = code
        self.iter = iter(_patched_opargs(_unpack_opargs(self.code.co_code)))

    def __iter__(self):
        return self

    def _fetch_opcode(self):
        return next(self.iter)

    def next(self):
        offset, opcode, arg, nextoffset = self._fetch_opcode()
        return (offset, ByteCodeInst(offset=offset, opcode=opcode, arg=arg, nextoffset=nextoffset))
    __next__ = next

    def read_arg(self, size):
        buf = 0
        for i in range(size):
            _offset, byte = next(self.iter)
            buf |= byte << 8 * i
        return buf