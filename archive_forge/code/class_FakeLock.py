import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
class FakeLock:
    """
    A stand-in for L{threading.Lock}.

    @ivar acquired: Whether this lock is presently acquired.
    """

    def __init__(self):
        """
        Create a lock in the un-acquired state.
        """
        self.acquired = False

    def acquire(self):
        """
        Acquire the lock.  Raise an exception if the lock is already acquired.
        """
        if self.acquired:
            raise WouldDeadlock()
        self.acquired = True

    def release(self):
        """
        Release the lock.  Raise an exception if the lock is not presently
        acquired.
        """
        if not self.acquired:
            raise ThreadError()
        self.acquired = False