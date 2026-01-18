from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class ProgramNameEvent(MetaEventWithText):
    """
    Program Name Event.

    """
    meta_command = 8
    length = 'variable'
    name = 'Program Name'