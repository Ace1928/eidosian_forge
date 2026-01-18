import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
class WouldDeadlock(Exception):
    """
    If this were a real lock, you'd be deadlocked because the lock would be
    double-acquired.
    """