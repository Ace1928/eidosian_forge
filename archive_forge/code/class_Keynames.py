import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
class Keynames(Enum):
    CURTSIES = auto()
    CURSES = auto()
    BYTES = auto()