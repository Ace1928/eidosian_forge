from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class IBodyProducer(IPushProducer):
    """
    Objects which provide L{IBodyProducer} write bytes to an object which
    provides L{IConsumer<twisted.internet.interfaces.IConsumer>} by calling its
    C{write} method repeatedly.

    L{IBodyProducer} providers may start producing as soon as they have an
    L{IConsumer<twisted.internet.interfaces.IConsumer>} provider.  That is, they
    should not wait for a C{resumeProducing} call to begin writing data.

    L{IConsumer.unregisterProducer<twisted.internet.interfaces.IConsumer.unregisterProducer>}
    must not be called.  Instead, the
    L{Deferred<twisted.internet.defer.Deferred>} returned from C{startProducing}
    must be fired when all bytes have been written.

    L{IConsumer.write<twisted.internet.interfaces.IConsumer.write>} may
    synchronously invoke any of C{pauseProducing}, C{resumeProducing}, or
    C{stopProducing}.  These methods must be implemented with this in mind.

    @since: 9.0
    """
    length = Attribute('\n        C{length} is a L{int} indicating how many bytes in total this\n        L{IBodyProducer} will write to the consumer or L{UNKNOWN_LENGTH}\n        if this is not known in advance.\n        ')

    def startProducing(consumer):
        """
        Start producing to the given
        L{IConsumer<twisted.internet.interfaces.IConsumer>} provider.

        @return: A L{Deferred<twisted.internet.defer.Deferred>} which stops
            production of data when L{Deferred.cancel} is called, and which
            fires with L{None} when all bytes have been produced or with a
            L{Failure<twisted.python.failure.Failure>} if there is any problem
            before all bytes have been produced.
        """

    def stopProducing():
        """
        In addition to the standard behavior of
        L{IProducer.stopProducing<twisted.internet.interfaces.IProducer.stopProducing>}
        (stop producing data), make sure the
        L{Deferred<twisted.internet.defer.Deferred>} returned by
        C{startProducing} is never fired.
        """