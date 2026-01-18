from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class TimeSignatureEvent(MetaEvent):
    """
    Time Signature Event.

    """
    meta_command = 88
    length = 4
    name = 'Time Signature'

    @property
    def numerator(self):
        """
        Numerator of the time signature.

        """
        return self.data[0]

    @numerator.setter
    def numerator(self, numerator):
        """
        Set numerator of the time signature.

        Parameters
        ----------
        numerator : int
            Numerator of the time signature.
        """
        self.data[0] = numerator

    @property
    def denominator(self):
        """
        Denominator of the time signature.

        """
        return 2 ** self.data[1]

    @denominator.setter
    def denominator(self, denominator):
        """
        Set denominator of the time signature.

        Parameters
        ----------
        denominator : int
            Denominator of the time signature.

        """
        self.data[1] = int(math.log(denominator, 2))

    @property
    def metronome(self):
        """
        Metronome.

        """
        return self.data[2]

    @metronome.setter
    def metronome(self, metronome):
        """
        Set metronome of the time signature.

        Parameters
        ----------
        metronome : int
            Metronome of the time signature.

        """
        self.data[2] = metronome

    @property
    def thirty_seconds(self):
        """
        Thirty-seconds of the time signature.

        """
        return self.data[3]

    @thirty_seconds.setter
    def thirty_seconds(self, thirty_seconds):
        """
        Set thirty-seconds of the time signature.

        Parameters
        ----------
        thirty_seconds : int
            Thirty-seconds of the time signature.

        """
        self.data[3] = thirty_seconds