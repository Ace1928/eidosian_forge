from typing import Callable
from zope.interface import implementer
from ._convenience import Quit
from ._ithreads import IExclusiveWorker
@implementer(IExclusiveWorker)
class ThreadWorker:
    """
    An L{IExclusiveWorker} implemented based on a single thread and a queue.

    This worker ensures exclusivity (i.e. it is an L{IExclusiveWorker} and not
    an L{IWorker}) by performing all of the work passed to C{do} on the I{same}
    thread.
    """

    def __init__(self, startThread, queue):
        """
        Create a L{ThreadWorker} with a function to start a thread and a queue
        to use to communicate with that thread.

        @param startThread: a callable that takes a callable to run in another
            thread.
        @type startThread: callable taking a 0-argument callable and returning
            nothing.

        @param queue: A L{Queue} to use to give tasks to the thread created by
            C{startThread}.
        @type queue: L{Queue}
        """
        self._q = queue
        self._hasQuit = Quit()

        def work():
            for task in iter(queue.get, _stop):
                task()
        startThread(work)

    def do(self, task: Callable[[], None]) -> None:
        """
        Perform the given task on the thread owned by this L{ThreadWorker}.

        @param task: the function to call on a thread.
        """
        self._hasQuit.check()
        self._q.put(task)

    def quit(self):
        """
        Reject all future work and stop the thread started by C{__init__}.
        """
        self._hasQuit.set()
        self._q.put(_stop)