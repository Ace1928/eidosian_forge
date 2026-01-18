from fixtures import Fixture
import signal
from typing import Union
from ._deferreddebug import DebugTwisted
from twisted.internet import defer
from twisted.internet.interfaces import IReactorThreads
from twisted.python.failure import Failure
from twisted.python.util import mergeFunctionMetadata
Run 'function' in a reactor.

        If 'function' returns a Deferred, the reactor will keep spinning until
        the Deferred fires and its chain completes or until the timeout is
        reached -- whichever comes first.

        :raise TimeoutError: If 'timeout' is reached before the Deferred
            returned by 'function' has completed its callback chain.
        :raise NoResultError: If the reactor is somehow interrupted before
            the Deferred returned by 'function' has completed its callback
            chain.
        :raise StaleJunkError: If there's junk in the spinner from a previous
            run.
        :return: Whatever is at the end of the function's callback chain.  If
            it's an error, then raise that.
        