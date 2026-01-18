import numpy as np
def _supported(self, a, b):
    """
        Return a subset of self containing the values that are in
        (or overlap with) the interval (a, b).
        """
    uncensored = self._uncensored
    uncensored = uncensored[(a < uncensored) & (uncensored < b)]
    left = self._left
    left = left[a < left]
    right = self._right
    right = right[right < b]
    interval = self._interval
    interval = interval[(a < interval[:, 1]) & (interval[:, 0] < b)]
    return CensoredData(uncensored, left=left, right=right, interval=interval)