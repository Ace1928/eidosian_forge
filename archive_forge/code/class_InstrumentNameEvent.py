from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class InstrumentNameEvent(MetaEventWithText):
    """
    Instrument Name Event.

    """
    meta_command = 4
    length = 'variable'
    name = 'Instrument Name'