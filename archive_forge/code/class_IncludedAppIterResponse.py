import warnings
from io import StringIO
class IncludedAppIterResponse(object):

    def __init__(self):
        self.status = None
        self.headers = None
        self.accumulated = []
        self.app_iter = None
        self._closed = False

    def close(self):
        assert not self._closed, 'Tried to close twice'
        if hasattr(self.app_iter, 'close'):
            self.app_iter.close()

    def write(self, s):
        self.accumulated.append