from time import time as gettime
def _putentry(self, key, entry):
    self._prunelowestweight()
    self._dict[key] = entry