import inspect
from functools import partial
from weakref import WeakMethod
def _remove_handler(self, name, handler):
    """Used internally to remove all handler instances for the given event name.

        This is normally called from a dead ``WeakMethod`` to remove itself from the
        event stack.
        """
    for frame in list(self._event_stack):
        if name in frame:
            try:
                if frame[name] == handler:
                    del frame[name]
                    if not frame:
                        self._event_stack.remove(frame)
            except TypeError:
                pass