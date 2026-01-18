from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class MetaEvent(Event):
    """
    MetaEvent is a special subclass of Event that is not meant to be used as a
    concrete class. It defines a subset of Events known as the Meta events.

    """
    status_msg = 255
    meta_command = 0
    name = 'Meta Event'

    def __eq__(self, other):
        return self.tick == other.tick and self.data == other.data and (self.status_msg == other.status_msg) and (self.meta_command == other.meta_command)