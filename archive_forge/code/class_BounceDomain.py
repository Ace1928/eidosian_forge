import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
@implementer(IDomain)
class BounceDomain:
    """
    A domain with no users.

    This can be used to block off a domain.
    """

    def exists(self, user):
        """
        Raise an exception to indicate that the user does not exist in this
        domain.

        @type user: L{User}
        @param user: A user.

        @raise SMTPBadRcpt: When the given user does not exist in this domain.
        """
        raise smtp.SMTPBadRcpt(user)

    def willRelay(self, user, protocol):
        """
        Indicate that this domain will not relay.

        @type user: L{Address}
        @param user: The destination address.

        @type protocol: L{Protocol <twisted.internet.protocol.Protocol>}
        @param protocol: The protocol over which the message to be relayed is
            being received.

        @rtype: L{bool}
        @return: C{False}.
        """
        return False

    def addUser(self, user, password):
        """
        Ignore attempts to add a user to this domain.

        @type user: L{bytes}
        @param user: A username.

        @type password: L{bytes}
        @param password: A password.
        """
        pass

    def getCredentialsCheckers(self):
        """
        Return no credentials checkers for this domain.

        @rtype: L{list}
        @return: The empty list.
        """
        return []