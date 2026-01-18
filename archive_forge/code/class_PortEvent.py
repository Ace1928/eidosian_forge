from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class PortEvent(MetaEvent):
    """
    Port Event.

    """
    meta_command = 33
    name = 'MIDI Port/Cable'