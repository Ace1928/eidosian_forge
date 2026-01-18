from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class PitchWheelEvent(ChannelEvent):
    """
    Pitch Wheel Event.

    """
    status_msg = 224
    length = 2
    name = 'Pitch Wheel'

    @property
    def pitch(self):
        """
        Pitch of the Pitch Wheel Event.

        """
        return (self.data[1] << 7 | self.data[0]) - 8192

    @pitch.setter
    def pitch(self, pitch):
        """
        Set the pitch of the Pitch Wheel Event.

        Parameters
        ----------
        pitch : int
            Pitch of the Pitch Wheel Event.

        """
        value = pitch + 8192
        self.data[0] = value & 127
        self.data[1] = value >> 7 & 127