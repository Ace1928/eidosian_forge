import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
def _sendFailureOrError(self, fail, requestID):
    """
        Call L{_sendError} or L{_sendFailure}, depending on whether C{fail}
        represents an L{Error} subclass or not.

        @param fail: The failure.
        @param requestID: The request ID.
        """
    if fail.check(Error) is None:
        self._sendFailure(fail, requestID)
    else:
        self._sendError(fail, requestID)