from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
class Interruption(Event):
    """Immediately schedules an :class:`~simpy.exceptions.Interrupt` exception
    with the given *cause* to be thrown into *process*.

    This event is automatically triggered when it is created.

    """

    def __init__(self, process: Process, cause: Optional[Any]):
        self.env = process.env
        self.callbacks: EventCallbacks = [self._interrupt]
        self._value = Interrupt(cause)
        self._ok = False
        self._defused = True
        if process._value is not PENDING:
            raise RuntimeError(f'{process} has terminated and cannot be interrupted.')
        if process is self.env.active_process:
            raise RuntimeError('A process is not allowed to interrupt itself.')
        self.process = process
        self.env.schedule(self, URGENT)

    def _interrupt(self, event: Event) -> None:
        if self.process._value is not PENDING:
            return
        self.process._target.callbacks.remove(self.process._resume)
        self.process._resume(self)