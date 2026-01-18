from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor

        Map a class id to a chord label.
        0..11 major chords, 12..23 minor chords, 24 no chord
        