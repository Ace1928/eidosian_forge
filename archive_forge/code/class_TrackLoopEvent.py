from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class TrackLoopEvent(MetaEvent):
    """
    Track Loop Event.

    """
    meta_command = 46
    name = 'Track Loop'