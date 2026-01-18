from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
def _remove_check_callbacks(self) -> None:
    """Remove _check() callbacks from events recursively.

        Once the condition has triggered, the condition's events no longer need
        to have _check() callbacks. Removing the _check() callbacks is
        important to break circular references between the condition and
        untriggered events.

        """
    for event in self._events:
        if event.callbacks and self._check in event.callbacks:
            event.callbacks.remove(self._check)
        if isinstance(event, Condition):
            event._remove_check_callbacks()