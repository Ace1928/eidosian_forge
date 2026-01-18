import logging
class HistoryRecorder:

    def __init__(self):
        self._enabled = False
        self._handlers = []

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def add_handler(self, handler):
        self._handlers.append(handler)

    def record(self, event_type, payload, source='BOTOCORE'):
        if self._enabled and self._handlers:
            for handler in self._handlers:
                try:
                    handler.emit(event_type, payload, source)
                except Exception:
                    logger.debug('Exception raised in %s.', handler, exc_info=True)