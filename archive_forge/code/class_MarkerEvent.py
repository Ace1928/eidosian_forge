from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class MarkerEvent(MetaEventWithText):
    """
    Marker Event.

    """
    meta_command = 6
    length = 'variable'
    name = 'Marker'