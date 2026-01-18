from zope.interface import Interface
class IAliasableDomain(IDomain):
    """
    An interface for email domains which can be aliased to other domains.
    """

    def setAliasGroup(aliases):
        """
        Set the group of defined aliases for this domain.

        @type aliases: L{dict} of L{bytes} -> L{IAlias} provider
        @param aliases: A mapping of domain name to alias.
        """

    def exists(user, memo=None):
        """
        Check whether a user exists in this domain or an alias of it.

        @type user: L{User}
        @param user: A user.

        @type memo: L{None} or L{dict} of
            L{AliasBase <twisted.mail.alias.AliasBase>}
        @param memo: A record of the addresses already considered while
            resolving aliases. The default value should be used by all external
            code.

        @rtype: no-argument callable which returns L{IMessageSMTP} provider
        @return: A function which takes no arguments and returns a message
            receiver for the user.

        @raise SMTPBadRcpt: When the given user does not exist in this domain
            or an alias of it.
        """