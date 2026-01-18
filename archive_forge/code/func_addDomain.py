import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def addDomain(self, name, domain):
    """
        Add a domain for which the service will accept email.

        @type name: L{bytes}
        @param name: A domain name.

        @type domain: L{IDomain} provider
        @param domain: A domain object.
        """
    portal = Portal(domain)
    map(portal.registerChecker, domain.getCredentialsCheckers())
    self.domains[name] = domain
    self.portals[name] = portal
    if self.aliases and IAliasableDomain.providedBy(domain):
        domain.setAliasGroup(self.aliases)