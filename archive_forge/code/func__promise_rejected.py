from functools import partial
def _promise_rejected(self, reason):
    if self.is_resolved:
        return False
    self._total_resolved += 1
    self._reject(reason)
    return True