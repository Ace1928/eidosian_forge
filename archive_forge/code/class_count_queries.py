from functools import wraps
import logging
class count_queries(object):

    def __init__(self, only_select=False):
        self.only_select = only_select
        self.count = 0

    def get_queries(self):
        return self._handler.queries

    def __enter__(self):
        self._handler = _QueryLogHandler()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.removeHandler(self._handler)
        if self.only_select:
            self.count = len([q for q in self._handler.queries if q.msg[0].startswith('SELECT ')])
        else:
            self.count = len(self._handler.queries)