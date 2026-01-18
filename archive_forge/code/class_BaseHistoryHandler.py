import logging
class BaseHistoryHandler:

    def emit(self, event_type, payload, source):
        raise NotImplementedError('emit()')