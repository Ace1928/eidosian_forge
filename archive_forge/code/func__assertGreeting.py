import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def _assertGreeting(self, user):
    """
        The user has been greeted with the four messages that are (usually)
        considered to start an IRC session.

        Asserts that the required responses were received.
        """
    response = self._response(user)
    expected = [irc.RPL_WELCOME, irc.RPL_YOURHOST, irc.RPL_CREATED, irc.RPL_MYINFO]
    for prefix, command, args in response:
        if command in expected:
            expected.remove(command)
    self.assertFalse(expected, f'Missing responses for {expected!r}')