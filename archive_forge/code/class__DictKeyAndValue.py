from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
class _DictKeyAndValue:

    def __init__(self, dict):
        self.dict = dict

    def __setitem__(self, n, obj):
        if n not in (1, 0):
            raise RuntimeError('DictKeyAndValue should only ever be called with 0 or 1')
        if n:
            self.value = obj
        else:
            self.key = obj
        if hasattr(self, 'key') and hasattr(self, 'value'):
            self.dict[self.key] = self.value