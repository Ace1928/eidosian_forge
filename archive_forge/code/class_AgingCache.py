from time import time as gettime
class AgingCache(BasicCache):
    """ This cache prunes out cache entries that are too old.
    """

    def __init__(self, maxentries=128, maxseconds=10.0):
        super(AgingCache, self).__init__(maxentries)
        self.maxseconds = maxseconds

    def _getentry(self, key):
        entry = self._dict[key]
        if entry.isexpired():
            self.delentry(key)
            raise KeyError(key)
        return entry

    def _build(self, key, builder):
        val = builder()
        entry = AgingEntry(val, gettime() + self.maxseconds)
        return entry