from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class ProgramChangeEvent(ChannelEvent):
    """
    Program Change Event.

    """
    status_msg = 192
    length = 1
    name = 'Program Change'

    def __str__(self):
        return '%s: tick: %s channel: %s value: %s' % (self.__class__.__name__, self.tick, self.channel, self.value)

    @property
    def value(self):
        """
        Value of the Program Change Event.

        """
        return self.data[0]

    @value.setter
    def value(self, value):
        """
        Set the value of the Program Change Event.

        Parameters
        ----------
        value : int
            Value of the Program Change Event.

        """
        self.data[0] = value