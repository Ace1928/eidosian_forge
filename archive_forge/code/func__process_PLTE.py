import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def _process_PLTE(self, data):
    if self.plte:
        warnings.warn('Multiple PLTE chunks present.')
    self.plte = data
    if len(data) % 3 != 0:
        raise FormatError("PLTE chunk's length should be a multiple of 3.")
    if len(data) > 2 ** self.bitdepth * 3:
        raise FormatError('PLTE chunk is too long.')
    if len(data) == 0:
        raise FormatError('Empty PLTE is not allowed.')