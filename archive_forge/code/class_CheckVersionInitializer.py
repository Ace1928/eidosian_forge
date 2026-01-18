from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
class CheckVersionInitializer:
    """
    Initializer that checks if the minimum common stream version number is 1.0.
    """

    def __init__(self, xs):
        self.xmlstream = xs

    def initialize(self):
        if self.xmlstream.version < (1, 0):
            raise error.StreamError('unsupported-version')