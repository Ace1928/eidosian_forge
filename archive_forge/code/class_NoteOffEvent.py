from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class NoteOffEvent(NoteEvent):
    """
    Note Off Event.

    """
    status_msg = 128
    name = 'Note Off'
    sort = 0.2