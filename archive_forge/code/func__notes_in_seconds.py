from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
def _notes_in_seconds(self, notes):
    """
        Converts onsets and offsets of notes from ticks to seconds.

        Parameters
        ----------
        notes : numpy array or list of tuples
            Notes (onset, pitch, offset, velocity).

        Returns
        -------
        notes : numpy array
            Notes with onset and offset times in seconds.

        """
    tempi = self.tempi(suppress_warnings=True)
    for note in notes:
        onset, _, offset, _, _ = note
        t_on = tempi[np.argmax(tempi[:, 0] > onset) - 1]
        t_off = tempi[np.argmax(tempi[:, 0] > offset) - 1]
        note[0] = (onset - t_on[0]) * t_on[1] + t_on[2]
        note[2] = (offset - t_off[0]) * t_off[1] + t_off[2]
    return notes