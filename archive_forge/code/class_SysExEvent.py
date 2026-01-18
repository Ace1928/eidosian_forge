from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class SysExEvent(Event):
    """
    System Exclusive Event.

    """
    status_msg = 240
    length = 'variable'
    name = 'SysEx'