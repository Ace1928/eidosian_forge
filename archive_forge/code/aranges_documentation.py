import os
from collections import namedtuple
from ..common.utils import struct_parse
from bisect import bisect_right
import math
 Given this set's header value (int) for the address size,
            get the Construct representation of that size
        