import bisect
class TopNContainer(object):
    """ maintains a sorted list of a particular number of data elements.

  """

    def __init__(self, size, mostNeg=-1e+99):
        """
    if size is negative, all entries will be kept in sorted order
    """
        self._size = size
        if size >= 0:
            self.best = [mostNeg] * self._size
            self.extras = [None] * self._size
        else:
            self.best = []
            self.extras = []

    def Insert(self, val, extra=None):
        """ only does the insertion if val fits """
        if self._size >= 0:
            if val > self.best[0]:
                idx = bisect.bisect(self.best, val)
                if idx == self._size:
                    self.best.append(val)
                    self.extras.append(extra)
                else:
                    self.best.insert(idx, val)
                    self.extras.insert(idx, extra)
                self.best.pop(0)
                self.extras.pop(0)
        else:
            idx = bisect.bisect(self.best, val)
            self.best.insert(idx, val)
            self.extras.insert(idx, extra)

    def GetPts(self):
        """ returns our set of points """
        return self.best

    def GetExtras(self):
        """ returns our set of extras """
        return self.extras

    def __len__(self):
        if self._size >= 0:
            return self._size
        else:
            return len(self.best)

    def __getitem__(self, which):
        return (self.best[which], self.extras[which])

    def reverse(self):
        self.best.reverse()
        self.extras.reverse()