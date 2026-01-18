from ..api import CacheBackend
from ..api import NO_VALUE
class NullLock(object):

    def acquire(self, wait=True):
        return True

    def release(self):
        pass

    def locked(self):
        return False