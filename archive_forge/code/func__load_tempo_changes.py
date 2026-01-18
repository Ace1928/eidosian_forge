from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
def _load_tempo_changes(self, midi_data):
    """Populates ``self._tick_scales`` with tuples of
        ``(tick, tick_scale)`` loaded from ``midi_data``.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        """
    self._tick_scales = [(0, 60.0 / (120.0 * self.resolution))]
    for event in midi_data.tracks[0]:
        if event.type == 'set_tempo':
            if event.time == 0:
                bpm = 60000000.0 / event.tempo
                self._tick_scales = [(0, 60.0 / (bpm * self.resolution))]
            else:
                _, last_tick_scale = self._tick_scales[-1]
                tick_scale = 60.0 / (60000000.0 / event.tempo * self.resolution)
                if tick_scale != last_tick_scale:
                    self._tick_scales.append((event.time, tick_scale))