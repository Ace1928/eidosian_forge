from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class ChannelAfterTouchEvent(ChannelEvent):
    """
    Channel After Touch Event.

    """
    status_msg = 208
    length = 1
    name = 'Channel After Touch'

    def __str__(self):
        return '%s: tick: %s channel: %s value: %s' % (self.__class__.__name__, self.tick, self.channel, self.value)

    @property
    def value(self):
        """
        Value of the Channel After Touch Event.

        """
        return self.data[0]

    @value.setter
    def value(self, value):
        """
        Set the value of the Channel After Touch Event.

        Parameters
        ----------
        value : int
            Value of the Channel After Touch Event.

        """
        self.data[0] = value