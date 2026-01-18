from zope.interface import implementer
from twisted.copyright import longversion
from twisted.cred.credentials import CramMD5Credentials, UsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol
from twisted.mail import pop3, relay, smtp
from twisted.python import log
@implementer(smtp.IMessageDelivery)
class DomainDeliveryBase:
    """
    A base class for message delivery using the domains of a mail service.

    @ivar service: See L{__init__}
    @ivar user: See L{__init__}
    @ivar host: See L{__init__}

    @type protocolName: L{bytes}
    @ivar protocolName: The protocol being used to deliver the mail.
        Sub-classes should set this appropriately.
    """
    service = None
    protocolName: bytes = b'not-implemented-protocol'

    def __init__(self, service, user, host=smtp.DNSNAME):
        """
        @type service: L{MailService}
        @param service: A mail service.

        @type user: L{bytes} or L{None}
        @param user: The authenticated SMTP user.

        @type host: L{bytes}
        @param host: The hostname.
        """
        self.service = service
        self.user = user
        self.host = host

    def receivedHeader(self, helo, origin, recipients):
        """
        Generate a received header string for a message.

        @type helo: 2-L{tuple} of (L{bytes}, L{bytes})
        @param helo: The client's identity as sent in the HELO command and its
            IP address.

        @type origin: L{Address}
        @param origin: The origination address of the message.

        @type recipients: L{list} of L{User}
        @param recipients: The destination addresses for the message.

        @rtype: L{bytes}
        @return: A received header string.
        """
        authStr = heloStr = b''
        if self.user:
            authStr = b' auth=' + self.user.encode('xtext')
        if helo[0]:
            heloStr = b' helo=' + helo[0]
        fromUser = b'from ' + helo[0] + b' ([' + helo[1] + b']' + heloStr + authStr
        by = b'by ' + self.host + b' with ' + self.protocolName + b' (' + longversion.encode('ascii') + b')'
        forUser = b'for <' + b' '.join(map(bytes, recipients)) + b'> ' + smtp.rfc822date()
        return b'Received: ' + fromUser + b'\n\t' + by + b'\n\t' + forUser

    def validateTo(self, user):
        """
        Validate the address for which a message is destined.

        @type user: L{User}
        @param user: The destination address.

        @rtype: L{Deferred <defer.Deferred>} which successfully fires with
            no-argument callable which returns L{IMessage <smtp.IMessage>}
            provider.
        @return: A deferred which successfully fires with a no-argument
            callable which returns a message receiver for the destination.

        @raise SMTPBadRcpt: When messages cannot be accepted for the
            destination address.
        """
        if self.user and self.service.queue:
            d = self.service.domains.get(user.dest.domain, None)
            if d is None:
                d = relay.DomainQueuer(self.service, True)
        else:
            d = self.service.domains[user.dest.domain]
        return defer.maybeDeferred(d.exists, user)

    def validateFrom(self, helo, origin):
        """
        Validate the address from which a message originates.

        @type helo: 2-L{tuple} of (L{bytes}, L{bytes})
        @param helo: The client's identity as sent in the HELO command and its
            IP address.

        @type origin: L{Address}
        @param origin: The origination address of the message.

        @rtype: L{Address}
        @return: The origination address.

        @raise SMTPBadSender: When messages cannot be accepted from the
            origination address.
        """
        if not helo:
            raise smtp.SMTPBadSender(origin, 503, 'Who are you?  Say HELO first.')
        if origin.local != b'' and origin.domain == b'':
            raise smtp.SMTPBadSender(origin, 501, 'Sender address must contain domain.')
        return origin