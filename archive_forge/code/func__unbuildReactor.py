import os
import signal
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Type, Union, cast
from zope.interface import Interface
from twisted.python import log
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
from twisted.python.failure import Failure
from twisted.python.reflect import namedAny
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, SynchronousTestCase
from twisted.trial.util import DEFAULT_TIMEOUT_DURATION, acquireAttribute
def _unbuildReactor(self, reactor):
    """
        Clean up any resources which may have been allocated for the given
        reactor by its creation or by a test which used it.
        """
    reactor._uninstallHandler()
    if getattr(reactor, '_internalReaders', None) is not None:
        for reader in reactor._internalReaders:
            reactor.removeReader(reader)
            reader.connectionLost(None)
        reactor._internalReaders.clear()
    reactor.disconnectAll()
    calls = reactor.getDelayedCalls()
    for c in calls:
        c.cancel()
    from twisted.internet import reactor as globalReactor
    globalReactor.__dict__ = reactor._originalReactorDict
    globalReactor.__class__ = reactor._originalReactorClass