from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class SmpteOffsetEvent(MetaEvent):
    """
    SMPTE Offset Event.

    """
    meta_command = 84
    name = 'SMPTE Offset'