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
def _process_sBIT(self, data):
    self.sbit = data
    if self.colormap and len(data) != 3 or (not self.colormap and len(data) != self.planes):
        raise FormatError('sBIT chunk has incorrect length.')