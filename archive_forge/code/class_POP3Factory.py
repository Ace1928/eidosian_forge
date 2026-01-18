from zope.interface import implementer
from twisted.copyright import longversion
from twisted.cred.credentials import CramMD5Credentials, UsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol
from twisted.mail import pop3, relay, smtp
from twisted.python import log
class POP3Factory(protocol.ServerFactory):
    """
    A POP3 server protocol factory.

    @ivar service: See L{__init__}

    @type protocol: no-argument callable which returns a L{Protocol
        <protocol.Protocol>} subclass
    @ivar protocol: A callable which creates a protocol.  The default value is
        L{VirtualPOP3}.
    """
    protocol = VirtualPOP3
    service = None

    def __init__(self, service):
        """
        @type service: L{MailService}
        @param service: An email service.
        """
        self.service = service

    def buildProtocol(self, addr):
        """
        Create an instance of a POP3 server protocol.

        @type addr: L{IAddress <twisted.internet.interfaces.IAddress>} provider
        @param addr: The address of the POP3 client.

        @rtype: L{POP3}
        @return: A POP3 protocol.
        """
        p = protocol.ServerFactory.buildProtocol(self, addr)
        p.service = self.service
        return p