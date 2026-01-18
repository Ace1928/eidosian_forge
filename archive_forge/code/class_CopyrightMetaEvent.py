from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class CopyrightMetaEvent(MetaEventWithText):
    """
    Copyright Meta Event.

    """
    meta_command = 2
    length = 'variable'
    name = 'Copyright Notice'