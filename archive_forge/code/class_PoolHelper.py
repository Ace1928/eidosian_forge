import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
class PoolHelper:
    """
    A L{PoolHelper} constructs a L{threadpool.ThreadPool} that doesn't actually
    use threads, by using the internal interfaces in L{twisted._threads}.

    @ivar performCoordination: a 0-argument callable that will perform one unit
        of "coordination" - work involved in delegating work to other threads -
        and return L{True} if it did any work, L{False} otherwise.

    @ivar workers: the workers which represent the threads within the pool -
        the workers other than the coordinator.
    @type workers: L{list} of 2-tuple of (L{IWorker}, C{workPerformer}) where
        C{workPerformer} is a 0-argument callable like C{performCoordination}.

    @ivar threadpool: a modified L{threadpool.ThreadPool} to test.
    @type threadpool: L{MemoryPool}
    """

    def __init__(self, testCase, *args, **kwargs):
        """
        Create a L{PoolHelper}.

        @param testCase: a test case attached to this helper.

        @type args: The arguments passed to a L{threadpool.ThreadPool}.

        @type kwargs: The arguments passed to a L{threadpool.ThreadPool}
        """
        coordinator, self.performCoordination = createMemoryWorker()
        self.workers = []

        def newWorker():
            self.workers.append(createMemoryWorker())
            return self.workers[-1][0]
        self.threadpool = MemoryPool(coordinator, testCase.fail, newWorker, *args, **kwargs)

    def performAllCoordination(self):
        """
        Perform all currently scheduled "coordination", which is the work
        involved in delegating work to other threads.
        """
        while self.performCoordination():
            pass