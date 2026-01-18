import collections
import enum
from . import exceptions
from . import mixins
from . import tasks
def _wake_up_next(self):
    """Wake up the first waiter that isn't done."""
    if not self._waiters:
        return
    for fut in self._waiters:
        if not fut.done():
            self._value -= 1
            fut.set_result(True)
            return