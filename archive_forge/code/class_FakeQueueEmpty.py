import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
class FakeQueueEmpty(Exception):
    """
    L{FakeQueue}'s C{get} has exhausted the queue.
    """