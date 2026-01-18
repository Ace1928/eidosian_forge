from functools import partial
def _promise_fulfilled(self, value, i):
    if self.is_resolved:
        return False
    self._values[i] = value
    self._total_resolved += 1
    if self._total_resolved >= self._length:
        self._resolve(self._values)
        return True
    return False