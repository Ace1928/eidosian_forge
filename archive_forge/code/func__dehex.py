from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def _dehex(s):
    """Liberally convert from hex string to binary string."""
    import re
    import binascii
    s = re.sub('[^a-fA-F\\d]', '', s)
    return binascii.unhexlify(strtobytes(s))