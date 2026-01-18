import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class SOffsetTFlags(Int32Flags):
    pass