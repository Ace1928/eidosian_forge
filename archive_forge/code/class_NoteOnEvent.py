from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class NoteOnEvent(NoteEvent):
    """
    Note On Event.

    """
    status_msg = 144
    name = 'Note On'
    sort = 0.1