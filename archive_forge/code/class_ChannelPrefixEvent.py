from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class ChannelPrefixEvent(MetaEvent):
    """
    Channel Prefix Event.

    """
    meta_command = 32
    length = 1
    name = 'Channel Prefix'