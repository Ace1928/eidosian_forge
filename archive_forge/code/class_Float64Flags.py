import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Float64Flags(object):
    bytewidth = 8
    min_val = None
    max_val = None
    py_type = float
    name = 'float64'
    packer_type = packer.float64