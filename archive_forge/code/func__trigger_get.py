from __future__ import annotations
from typing import (
from simpy.core import BoundClass, Environment
from simpy.events import Event, Process
def _trigger_get(self, put_event: Optional[PutType]) -> None:
    """Trigger get events.

        This method is called once a new get event has been created or a put
        event has been processed.

        The method iterates over all get events in the :attr:`get_queue` and
        calls :meth:`_do_get` to check if the conditions for the event are met.
        If :meth:`_do_get` returns ``False``, the iteration is stopped early.
        """
    idx = 0
    while idx < len(self.get_queue):
        get_event = self.get_queue[idx]
        proceed = self._do_get(get_event)
        if not get_event.triggered:
            idx += 1
        elif self.get_queue.pop(idx) != get_event:
            raise RuntimeError('Get queue invariant violated')
        if not proceed:
            break