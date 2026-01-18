import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def _wrappedLogPrefix(wrapper, wrapped):
    """
    Compute a log prefix for a wrapper and the object it wraps.

    @rtype: C{str}
    """
    if ILoggingContext.providedBy(wrapped):
        logPrefix = wrapped.logPrefix()
    else:
        logPrefix = wrapped.__class__.__name__
    return f'{logPrefix} ({wrapper.__class__.__name__})'