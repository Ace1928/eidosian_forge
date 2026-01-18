from zope.interface import implementer
from twisted.copyright import longversion
from twisted.cred.credentials import CramMD5Credentials, UsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol
from twisted.mail import pop3, relay, smtp
from twisted.python import log
class ESMTPFactory(SMTPFactory):
    """
    An ESMTP server protocol factory.

    @type protocol: no-argument callable which returns a L{Protocol
        <protocol.Protocol>} subclass
    @ivar protocol: A callable which creates a protocol.  The default value is
        L{ESMTP}.

    @type context: L{IOpenSSLContextFactory
        <twisted.internet.interfaces.IOpenSSLContextFactory>} or L{None}
    @ivar context: A factory to generate contexts to be used in negotiating
        encrypted communication.

    @type challengers: L{dict} mapping L{bytes} to no-argument callable which
        returns L{ICredentials <twisted.cred.credentials.ICredentials>}
        subclass provider.
    @ivar challengers: A mapping of acceptable authorization mechanism to
        callable which creates credentials to use for authentication.
    """
    protocol = smtp.ESMTP
    context = None

    def __init__(self, *args):
        """
        @param args: Arguments for L{SMTPFactory.__init__}

        @see: L{SMTPFactory.__init__}
        """
        SMTPFactory.__init__(self, *args)
        self.challengers = {b'CRAM-MD5': CramMD5Credentials}

    def buildProtocol(self, addr):
        """
        Create an instance of an ESMTP server protocol.

        @type addr: L{IAddress <twisted.internet.interfaces.IAddress>} provider
        @param addr: The address of the ESMTP client.

        @rtype: L{ESMTP}
        @return: An ESMTP protocol.
        """
        p = SMTPFactory.buildProtocol(self, addr)
        p.challengers = self.challengers
        p.ctx = self.context
        return p