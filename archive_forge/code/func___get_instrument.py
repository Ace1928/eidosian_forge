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
def __get_instrument(program, channel, track, create_new):
    """Gets the Instrument corresponding to the given program number,
            drum/non-drum type, channel, and track index.  If no such
            instrument exists, one is created.

            """
    if (program, channel, track) in instrument_map:
        return instrument_map[program, channel, track]
    if not create_new and (channel, track) in stragglers:
        return stragglers[channel, track]
    if create_new:
        is_drum = channel == 9
        instrument = Instrument(program, is_drum, track_name_map[track_idx])
        if (channel, track) in stragglers:
            straggler = stragglers[channel, track]
            instrument.control_changes = straggler.control_changes
            instrument.pitch_bends = straggler.pitch_bends
        instrument_map[program, channel, track] = instrument
    else:
        instrument = Instrument(program, track_name_map[track_idx])
        stragglers[channel, track] = instrument
    return instrument