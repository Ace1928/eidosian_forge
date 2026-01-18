import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
class RacyLockWorker(LockWorker):

    @property
    def _lock(self):
        self.quit()
        return self.__dict__['_lock']

    @_lock.setter
    def _lock(self, value):
        self.__dict__['_lock'] = value