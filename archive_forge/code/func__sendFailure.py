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
def _sendFailure(self, fail, requestID):
    """
        Log error and then send it.

        @param fail: The failure.
        @param requestID: The request ID.
        """
    log.msg('Peer will receive following PB traceback:')
    log.err(fail)
    self._sendError(fail, requestID)