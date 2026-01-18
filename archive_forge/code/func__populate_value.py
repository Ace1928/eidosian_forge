from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
def _populate_value(self, value: ConditionValue) -> None:
    """Populate the *value* by recursively visiting all nested
        conditions."""
    for event in self._events:
        if isinstance(event, Condition):
            event._populate_value(value)
        elif event.callbacks is None:
            value.events.append(event)