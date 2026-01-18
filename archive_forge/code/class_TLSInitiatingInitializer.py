from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
class TLSInitiatingInitializer(BaseFeatureInitiatingInitializer):
    """
    TLS stream initializer for the initiating entity.

    It is strongly required to include this initializer in the list of
    initializers for an XMPP stream. By default it will try to negotiate TLS.
    An XMPP server may indicate that TLS is required. If TLS is not desired,
    set the C{wanted} attribute to False instead of removing it from the list
    of initializers, so a proper exception L{TLSRequired} can be raised.

    @ivar wanted: indicates if TLS negotiation is wanted.
    @type wanted: C{bool}
    """
    feature = (NS_XMPP_TLS, 'starttls')
    wanted = True
    _deferred = None
    _configurationForTLS = None

    def __init__(self, xs, required=True, configurationForTLS=None):
        """
        @param configurationForTLS: An object which creates appropriately
            configured TLS connections. This is passed to C{startTLS} on the
            transport and is preferably created using
            L{twisted.internet.ssl.optionsForClientTLS}.  If C{None}, the
            default is to verify the server certificate against the trust roots
            as provided by the platform. See
            L{twisted.internet._sslverify.platformTrust}.
        @type configurationForTLS: L{IOpenSSLClientConnectionCreator} or
            C{None}
        """
        super().__init__(xs, required=required)
        self._configurationForTLS = configurationForTLS

    def onProceed(self, obj):
        """
        Proceed with TLS negotiation and reset the XML stream.
        """
        self.xmlstream.removeObserver('/failure', self.onFailure)
        if self._configurationForTLS:
            ctx = self._configurationForTLS
        else:
            ctx = ssl.optionsForClientTLS(self.xmlstream.otherEntity.host)
        self.xmlstream.transport.startTLS(ctx)
        self.xmlstream.reset()
        self.xmlstream.sendHeader()
        self._deferred.callback(Reset)

    def onFailure(self, obj):
        self.xmlstream.removeObserver('/proceed', self.onProceed)
        self._deferred.errback(TLSFailed())

    def start(self):
        """
        Start TLS negotiation.

        This checks if the receiving entity requires TLS, the SSL library is
        available and uses the C{required} and C{wanted} instance variables to
        determine what to do in the various different cases.

        For example, if the SSL library is not available, and wanted and
        required by the user, it raises an exception. However if it is not
        required by both parties, initialization silently succeeds, moving
        on to the next step.
        """
        if self.wanted:
            if ssl is None:
                if self.required:
                    return defer.fail(TLSNotSupported())
                else:
                    return defer.succeed(None)
            else:
                pass
        elif self.xmlstream.features[self.feature].required:
            return defer.fail(TLSRequired())
        else:
            return defer.succeed(None)
        self._deferred = defer.Deferred()
        self.xmlstream.addOnetimeObserver('/proceed', self.onProceed)
        self.xmlstream.addOnetimeObserver('/failure', self.onFailure)
        self.xmlstream.send(domish.Element((NS_XMPP_TLS, 'starttls')))
        return self._deferred