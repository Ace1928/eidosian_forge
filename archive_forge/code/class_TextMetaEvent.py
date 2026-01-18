from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class TextMetaEvent(MetaEventWithText):
    """
    Text Meta Event.

    """
    meta_command = 1
    length = 'variable'
    name = 'Text'