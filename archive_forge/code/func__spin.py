import logging
import threading
from grpc.framework.foundation import stream
def _spin(self, sink, value, terminate):
    while True:
        try:
            if value is _NO_VALUE:
                sink.terminate()
            elif terminate:
                sink.consume_and_terminate(value)
            else:
                sink.consume(value)
        except Exception as e:
            _LOGGER.exception(e)
        with self._lock:
            if terminate:
                self._spinning = False
                return
            elif self._values:
                value = self._values.pop(0)
                terminate = not self._values and (not self._active)
            elif not self._active:
                value = _NO_VALUE
                terminate = True
            else:
                self._spinning = False
                return