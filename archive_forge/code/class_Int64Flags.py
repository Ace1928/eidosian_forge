import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Int64Flags(object):
    bytewidth = 8
    min_val = -2 ** 63
    max_val = 2 ** 63 - 1
    py_type = int
    name = 'int64'
    packer_type = packer.int64