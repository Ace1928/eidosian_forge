import collections
import random
import threading
def Keys(self):
    """Return all the keys in the reservoir.

        Returns:
          ['list', 'of', 'keys'] in the Reservoir.
        """
    with self._mutex:
        return list(self._buckets.keys())