from zope.interface import implementer
from twisted.copyright import longversion
from twisted.cred.credentials import CramMD5Credentials, UsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol
from twisted.mail import pop3, relay, smtp
from twisted.python import log
class SMTPDomainDelivery(DomainDeliveryBase):
    """
    A domain delivery base class for use in an SMTP server.
    """
    protocolName = b'smtp'