from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def addCallback(self, fn, *args, **kwargs):
    """
        Register a callback for notification when the IQ result is available.
        """
    self.callbacks.addCallback(True, fn, *args, **kwargs)