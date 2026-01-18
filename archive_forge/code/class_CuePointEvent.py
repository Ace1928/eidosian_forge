from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class CuePointEvent(MetaEventWithText):
    """
    Cue Point Event.

    """
    meta_command = 7
    length = 'variable'
    name = 'Cue Point'