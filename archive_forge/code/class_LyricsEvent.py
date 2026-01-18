from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class LyricsEvent(MetaEventWithText):
    """
    Lyrics Event.

    """
    meta_command = 5
    length = 'variable'
    name = 'Lyrics'