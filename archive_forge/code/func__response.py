import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def _response(self, user, messageType=None):
    """
        Extracts the user's response, and returns a list of parsed lines.
        If messageType is defined, only messages of that type will be returned.
        """
    response = user.transport.value()
    if bytes != str and isinstance(response, bytes):
        response = response.decode('utf-8')
    response = response.splitlines()
    user.transport.clear()
    result = []
    for message in map(irc.parsemsg, response):
        if messageType is None or message[1] == messageType:
            result.append(message)
    return result