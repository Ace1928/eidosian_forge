import time
from typing import Sequence
import curtsies.events
class UndoEvent(curtsies.events.Event):
    """Request to undo."""

    def __init__(self, n: int=1) -> None:
        self.n = n