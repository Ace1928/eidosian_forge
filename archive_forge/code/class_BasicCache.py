from time import time as gettime
class BasicCache(object):

    def __init__(self, maxentries=128):
        self.maxentries = maxentries
        self.prunenum = int(maxentries - maxentries / 8)
        self._dict = {}

    def clear(self):
        self._dict.clear()

    def _getentry(self, key):
        return self._dict[key]

    def _putentry(self, key, entry):
        self._prunelowestweight()
        self._dict[key] = entry

    def delentry(self, key, raising=False):
        try:
            del self._dict[key]
        except KeyError:
            if raising:
                raise

    def getorbuild(self, key, builder):
        try:
            entry = self._getentry(key)
        except KeyError:
            entry = self._build(key, builder)
            self._putentry(key, entry)
        return entry.value

    def _prunelowestweight(self):
        """ prune out entries with lowest weight. """
        numentries = len(self._dict)
        if numentries >= self.maxentries:
            items = [(entry.weight, key) for key, entry in self._dict.items()]
            items.sort()
            index = numentries - self.prunenum
            if index > 0:
                for weight, key in items[:index]:
                    self.delentry(key, raising=False)