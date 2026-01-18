import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _deprecateReactor(self, reactor):
    """
        Deprecate C{iterate}, C{crash} and C{stop} on C{reactor}. That is,
        each method is wrapped in a function that issues a deprecation
        warning, then calls the original.

        @param reactor: The Twisted reactor.
        """
    self._reactorMethods = {}
    for name in ['crash', 'iterate', 'stop']:
        self._reactorMethods[name] = getattr(reactor, name)
        setattr(reactor, name, self._makeReactorMethod(name))