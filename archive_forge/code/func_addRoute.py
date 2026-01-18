from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def addRoute(self, destination, xs):
    """
        Add a new route.

        The passed XML Stream C{xs} will have an observer for all stanzas
        added to route its outgoing traffic. In turn, traffic for
        C{destination} will be passed to this stream.

        @param destination: Destination of the route to be added as a host name
                            or L{None} for the default route.
        @type destination: C{str} or L{None}.
        @param xs: XML Stream to register the route for.
        @type xs: L{EventDispatcher<utility.EventDispatcher>}.
        """
    self.routes[destination] = xs
    xs.addObserver('/*', self.route)