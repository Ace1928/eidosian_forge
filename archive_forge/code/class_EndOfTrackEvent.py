from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class EndOfTrackEvent(MetaEvent):
    """
    End Of Track Event.

    """
    meta_command = 47
    name = 'End of Track'
    sort = 0.99