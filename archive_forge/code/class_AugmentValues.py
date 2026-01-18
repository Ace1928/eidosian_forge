import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
@dataclass
class AugmentValues:
    instrument_bin_remap: Dict[int, int]
    velocity_mod_factor: float
    transpose_semitones: int
    time_stretch_factor: float

    @classmethod
    def default(cls) -> 'AugmentValues':
        return cls(instrument_bin_remap={}, velocity_mod_factor=1.0, transpose_semitones=0, time_stretch_factor=1.0)