from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class KeySignatureEvent(MetaEvent):
    """
    Key Signature Event.

    """
    meta_command = 89
    length = 2
    name = 'Key Signature'

    @property
    def alternatives(self):
        """
        Alternatives of the key signature.

        """
        return self.data[0] - 256 if self.data[0] > 127 else self.data[0]

    @alternatives.setter
    def alternatives(self, alternatives):
        """
        Set alternatives of the key signature.

        Parameters
        ----------
        alternatives : int
            Alternatives of the key signature.

        """
        self.data[0] = 256 + alternatives if alternatives < 0 else alternatives

    @property
    def minor(self):
        """
        Major / minor.

        """
        return self.data[1]

    @minor.setter
    def minor(self, val):
        """
        Set major / minor.

        Parameters
        ----------
        val : int
            Major / minor.

        """
        self.data[1] = val