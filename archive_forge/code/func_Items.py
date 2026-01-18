import collections
import random
import threading
def Items(self):
    """Get all the items in the bucket."""
    with self._mutex:
        return list(self.items)