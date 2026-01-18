from functools import partial
def _init_promise(self, values):
    if values.is_fulfilled:
        values = values._value()
    elif values.is_rejected:
        self._reject(values._reason())
        return
    self.promise._is_async_guaranteed = True
    values._then(self._init, self._reject)
    return