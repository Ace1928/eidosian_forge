import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
class ScheduledEvent(Event):
    """Event scheduled for a future time.

    args:
        when (float): unix time in seconds for which this event is scheduled

    Custom events that occur at a specific time in the future should
    be subclassed from ScheduledEvent."""

    def __init__(self, when: float) -> None:
        self.when = when