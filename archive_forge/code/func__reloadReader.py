import threading
from tensorboard import errors
def _reloadReader(self):
    """If a reader exists and has started period updating, unblock the update.

        The updates are performed periodically with a sleep interval between
        successive calls to the reader's update() method. Calling this method
        interrupts the sleep immediately if one is ongoing.
        """
    if self._reload_needed_event:
        self._reload_needed_event.set()