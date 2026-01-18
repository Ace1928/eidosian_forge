from __future__ import annotations
from typing import (
from simpy.core import BoundClass, Environment
from simpy.events import Event, Process
def _trigger_put(self, get_event: Optional[GetType]) -> None:
    """This method is called once a new put event has been created or a get
        event has been processed.

        The method iterates over all put events in the :attr:`put_queue` and
        calls :meth:`_do_put` to check if the conditions for the event are met.
        If :meth:`_do_put` returns ``False``, the iteration is stopped early.
        """
    idx = 0
    while idx < len(self.put_queue):
        put_event = self.put_queue[idx]
        proceed = self._do_put(put_event)
        if not put_event.triggered:
            idx += 1
        elif self.put_queue.pop(idx) != put_event:
            raise RuntimeError('Put queue invariant violated')
        if not proceed:
            break