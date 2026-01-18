from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class ControlChangeEvent(ChannelEvent):
    """
    Control Change Event.

    """
    status_msg = 176
    length = 2
    name = 'Control Change'

    def __str__(self):
        return '%s: tick: %s channel: %s control: %s value: %s' % (self.__class__.__name__, self.tick, self.channel, self.control, self.value)

    @property
    def control(self):
        """
        Control ID.

        """
        return self.data[0]

    @control.setter
    def control(self, control):
        """
        Set control ID.

        Parameters
        ----------
        control : int
            Control ID.

        """
        self.data[0] = control

    @property
    def value(self):
        """
        Value of the controller.

        """
        return self.data[1]

    @value.setter
    def value(self, value):
        """
        Set the value of the controller.

        Parameters
        ----------
        value : int
            Value of the controller.

        """
        self.data[1] = value