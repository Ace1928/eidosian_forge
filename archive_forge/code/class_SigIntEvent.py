import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
class SigIntEvent(Event):
    """Event signifying a SIGINT"""

    def __repr__(self) -> str:
        return '<SigInt Event>'

    @property
    def name(self) -> str:
        return repr(self)