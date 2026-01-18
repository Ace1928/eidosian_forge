import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Uint32Flags(object):
    bytewidth = 4
    min_val = 0
    max_val = 2 ** 32 - 1
    py_type = int
    name = 'uint32'
    packer_type = packer.uint32