import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _blocked(self, f, *a):
    """
        Block a command, if necessary.

        If commands are being blocked, append information about the function
        which sends the command to a list and return a deferred that will be
        chained with the return value of the function when it eventually runs.
        Otherwise, set up for subsequent commands to be blocked and return
        L{None}.

        @type f: callable
        @param f: A function which sends a command.

        @type a: L{tuple}
        @param a: Arguments to the function.

        @rtype: L{None} or L{Deferred <defer.Deferred>}
        @return: L{None} if the command can run immediately.  Otherwise,
            a deferred that will eventually trigger with the return value of
            the function.
        """
    if self._blockedQueue is not None:
        d = defer.Deferred()
        self._blockedQueue.append((d, f, a))
        return d
    self._blockedQueue = []
    return None