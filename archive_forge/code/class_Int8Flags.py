import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Int8Flags(object):
    bytewidth = 1
    min_val = -2 ** 7
    max_val = 2 ** 7 - 1
    py_type = int
    name = 'int8'
    packer_type = packer.int8