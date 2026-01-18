import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
@dataclass
class DecodeState:
    total_time: float
    delta_accum: float
    current_bin: int
    current_note: int
    active_notes: Dict[Tuple[int, int], float]