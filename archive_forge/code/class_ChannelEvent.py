from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class ChannelEvent(Event):
    """
    Event with a channel number.

    """
    name = 'ChannelEvent'

    def __init__(self, **kwargs):
        super(ChannelEvent, self).__init__(**kwargs)
        self.channel = kwargs.get('channel', 0)

    def __eq__(self, other):
        return self.tick == other.tick and self.channel == other.channel and (self.data == other.data) and (self.status_msg == other.status_msg)

    def __str__(self):
        return '%s: tick: %s channel: %s data: %s' % (self.__class__.__name__, self.tick, self.channel, self.data)