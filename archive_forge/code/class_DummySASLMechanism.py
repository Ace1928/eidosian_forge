from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
@implementer(sasl_mechanisms.ISASLMechanism)
class DummySASLMechanism:
    """
    Dummy SASL mechanism.

    This just returns the initialResponse passed on creation, stores any
    challenges and replies with the value of C{response}.

    @ivar challenge: Last received challenge.
    @type challenge: C{unicode}.
    @ivar initialResponse: Initial response to be returned when requested
                           via C{getInitialResponse} or L{None}.
    @type initialResponse: C{unicode}
    """
    challenge = None
    name = 'DUMMY'
    response = b''

    def __init__(self, initialResponse):
        self.initialResponse = initialResponse

    def getInitialResponse(self):
        return self.initialResponse

    def getResponse(self, challenge):
        self.challenge = challenge
        return self.response