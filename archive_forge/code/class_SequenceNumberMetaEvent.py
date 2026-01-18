from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class SequenceNumberMetaEvent(MetaEvent):
    """
    Sequence Number Meta Event.

    """
    meta_command = 0
    length = 2
    name = 'Sequence Number'