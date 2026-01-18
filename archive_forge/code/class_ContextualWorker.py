from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
class ContextualWorker(proxyForInterface(IWorker, '_realWorker')):
    """
    A worker implementation that supplies a context.
    """

    def __init__(self, realWorker, **ctx):
        """
        Create with a real worker and a context.
        """
        self._realWorker = realWorker
        self._context = ctx

    def do(self, work):
        """
        Perform the given work with the context given to __init__.

        @param work: the work to pass on to the real worker.
        """
        super().do(lambda: call(self._context, work))