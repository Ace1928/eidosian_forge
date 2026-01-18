from zope.interface import implementer
from twisted.copyright import longversion
from twisted.cred.credentials import CramMD5Credentials, UsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol
from twisted.mail import pop3, relay, smtp
from twisted.python import log
class VirtualPOP3(pop3.POP3):
    """
    A virtual hosting POP3 server.

    @type service: L{MailService}
    @ivar service: The email service that created this server.  This must be
        set by the service.

    @type domainSpecifier: L{bytes}
    @ivar domainSpecifier: The character to use to split an email address into
        local-part and domain. The default is '@'.
    """
    service = None
    domainSpecifier = b'@'

    def authenticateUserAPOP(self, user, digest):
        """
        Perform APOP authentication.

        Override the default lookup scheme to allow virtual domains.

        @type user: L{bytes}
        @param user: The name of the user attempting to log in.

        @type digest: L{bytes}
        @param digest: The challenge response.

        @rtype: L{Deferred} which successfully results in 3-L{tuple} of
            (L{IMailbox <pop3.IMailbox>}, L{IMailbox <pop3.IMailbox>}
            provider, no-argument callable)
        @return: A deferred which fires when authentication is complete.
            If successful, it returns an L{IMailbox <pop3.IMailbox>} interface,
            a mailbox and a logout function. If authentication fails, the
            deferred fails with an L{UnauthorizedLogin
            <twisted.cred.error.UnauthorizedLogin>} error.
        """
        user, domain = self.lookupDomain(user)
        try:
            portal = self.service.lookupPortal(domain)
        except KeyError:
            return defer.fail(UnauthorizedLogin())
        else:
            return portal.login(pop3.APOPCredentials(self.magic, user, digest), None, pop3.IMailbox)

    def authenticateUserPASS(self, user, password):
        """
        Perform authentication for a username/password login.

        Override the default lookup scheme to allow virtual domains.

        @type user: L{bytes}
        @param user: The name of the user attempting to log in.

        @type password: L{bytes}
        @param password: The password to authenticate with.

        @rtype: L{Deferred} which successfully results in 3-L{tuple} of
            (L{IMailbox <pop3.IMailbox>}, L{IMailbox <pop3.IMailbox>}
            provider, no-argument callable)
        @return: A deferred which fires when authentication is complete.
            If successful, it returns an L{IMailbox <pop3.IMailbox>} interface,
            a mailbox and a logout function. If authentication fails, the
            deferred fails with an L{UnauthorizedLogin
            <twisted.cred.error.UnauthorizedLogin>} error.
        """
        user, domain = self.lookupDomain(user)
        try:
            portal = self.service.lookupPortal(domain)
        except KeyError:
            return defer.fail(UnauthorizedLogin())
        else:
            return portal.login(UsernamePassword(user, password), None, pop3.IMailbox)

    def lookupDomain(self, user):
        """
        Check whether a domain is among the virtual domains supported by the
        mail service.

        @type user: L{bytes}
        @param user: An email address.

        @rtype: 2-L{tuple} of (L{bytes}, L{bytes})
        @return: The local part and the domain part of the email address if the
            domain is supported.

        @raise POP3Error: When the domain is not supported by the mail service.
        """
        try:
            user, domain = user.split(self.domainSpecifier, 1)
        except ValueError:
            domain = b''
        if domain not in self.service.domains:
            raise pop3.POP3Error('no such domain {}'.format(domain.decode('utf-8')))
        return (user, domain)