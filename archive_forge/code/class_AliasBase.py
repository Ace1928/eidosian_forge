import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
class AliasBase:
    """
    The default base class for aliases.

    @ivar domains: See L{__init__}.

    @type original: L{Address}
    @ivar original: The original address being aliased.
    """

    def __init__(self, domains, original):
        """
        @type domains: L{dict} mapping L{bytes} to L{IDomain} provider
        @param domains: A mapping of domain name to domain object.

        @type original: L{bytes}
        @param original: The original address being aliased.
        """
        self.domains = domains
        self.original = smtp.Address(original)

    def domain(self):
        """
        Return the domain associated with original address.

        @rtype: L{IDomain} provider
        @return: The domain for the original address.
        """
        return self.domains[self.original.domain]

    def resolve(self, aliasmap, memo=None):
        """
        Map this alias to its ultimate destination.

        @type aliasmap: L{dict} mapping L{bytes} to L{AliasBase}
        @param aliasmap: A mapping of username to alias or group of aliases.

        @type memo: L{None} or L{dict} of L{AliasBase}
        @param memo: A record of the aliases already considered in the
            resolution process.  If provided, C{memo} is modified to include
            this alias.

        @rtype: L{IMessage <smtp.IMessage>} or L{None}
        @return: A message receiver for the ultimate destination or None for
            an invalid destination.
        """
        if memo is None:
            memo = {}
        if str(self) in memo:
            return None
        memo[str(self)] = None
        return self.createMessageReceiver()