from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
def _notes_in_beats(self, notes):
    """
        Converts onsets and offsets of notes from ticks to beats.

        Parameters
        ----------
        notes : numpy array or list of tuples
            Notes (onset, pitch, offset, velocity).

        Returns
        -------
        notes : numpy array
            Notes with onsets and offsets in beats.

        """
    tpq = self.ticks_per_quarter_note
    time_signatures = self.time_signatures(suppress_warnings=True)
    time_signatures[0, 1] = 0
    qnbtsc = np.diff(time_signatures[:, 0]) / tpq
    bbtsc = qnbtsc * (time_signatures[:-1, 2] / 4.0)
    time_signatures[1:, 1] = bbtsc.cumsum()
    for note in notes:
        onset, _, offset, _, _ = note
        tsc = time_signatures[np.argmax(time_signatures[:, 0] > onset) - 1]
        onset_ticks_since_tsc = onset - tsc[0]
        note[0] = tsc[1] + onset_ticks_since_tsc / tpq * (tsc[2] / 4.0)
        offset_ticks_since_tsc = offset - tsc[0]
        note[2] = tsc[1] + offset_ticks_since_tsc / tpq * (tsc[2] / 4.0)
    return notes