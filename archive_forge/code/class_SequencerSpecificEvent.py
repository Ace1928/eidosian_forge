from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class SequencerSpecificEvent(MetaEvent):
    """
    Sequencer Specific Event.

    """
    meta_command = 127
    name = 'Sequencer Specific'