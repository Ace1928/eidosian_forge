import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Uint16Flags(object):
    bytewidth = 2
    min_val = 0
    max_val = 2 ** 16 - 1
    py_type = int
    name = 'uint16'
    packer_type = packer.uint16