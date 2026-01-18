from time import time as gettime
def _getentry(self, key):
    entry = self._dict[key]
    if entry.isexpired():
        self.delentry(key)
        raise KeyError(key)
    return entry