import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
@implementer(IAlias)
class AliasGroup(AliasBase):
    """
    An alias which points to multiple destination aliases.

    @type processAliasFactory: no-argument callable which returns
        L{ProcessAlias}
    @ivar processAliasFactory: A factory for process aliases.

    @type aliases: L{list} of L{AliasBase} which implements L{IAlias}
    @ivar aliases: The destination aliases.
    """
    processAliasFactory = ProcessAlias

    def __init__(self, items, *args):
        """
        Create a group of aliases.

        Parse a list of alias strings and, for each, create an appropriate
        alias object.

        @type items: L{list} of L{bytes}
        @param items: Aliases.

        @type args: n-L{tuple} of (0) L{dict} mapping L{bytes} to L{IDomain}
            provider, (1) L{bytes}
        @param args: Arguments for L{AliasBase.__init__}.
        """
        AliasBase.__init__(self, *args)
        self.aliases = []
        while items:
            addr = items.pop().strip()
            if addr.startswith(':'):
                try:
                    f = open(addr[1:])
                except BaseException:
                    log.err(f'Invalid filename in alias file {addr[1:]!r}')
                else:
                    with f:
                        addr = ' '.join([l.strip() for l in f])
                    items.extend(addr.split(','))
            elif addr.startswith('|'):
                self.aliases.append(self.processAliasFactory(addr[1:], *args))
            elif addr.startswith('/'):
                if os.path.isdir(addr):
                    log.err('Directory delivery not supported')
                else:
                    self.aliases.append(FileAlias(addr, *args))
            else:
                self.aliases.append(AddressAlias(addr, *args))

    def __len__(self):
        """
        Return the number of aliases in the group.

        @rtype: L{int}
        @return: The number of aliases in the group.
        """
        return len(self.aliases)

    def __str__(self) -> str:
        """
        Build a string representation of this L{AliasGroup} instance.

        @rtype: L{bytes}
        @return: A string containing the aliases in the group.
        """
        return '<AliasGroup [%s]>' % ', '.join(map(str, self.aliases))

    def createMessageReceiver(self):
        """
        Create a message receiver for each alias and return a message receiver
        which will pass on a message to each of those.

        @rtype: L{MultiWrapper}
        @return: A message receiver which passes a message on to message
            receivers for each alias in the group.
        """
        return MultiWrapper([a.createMessageReceiver() for a in self.aliases])

    def resolve(self, aliasmap, memo=None):
        """
        Map each of the aliases in the group to its ultimate destination.

        @type aliasmap: L{dict} mapping L{bytes} to L{AliasBase}
        @param aliasmap: A mapping of username to alias or group of aliases.

        @type memo: L{None} or L{dict} of L{AliasBase}
        @param memo: A record of the aliases already considered in the
            resolution process.  If provided, C{memo} is modified to include
            this alias.

        @rtype: L{MultiWrapper}
        @return: A message receiver which passes the message on to message
            receivers for the ultimate destination of each alias in the group.
        """
        if memo is None:
            memo = {}
        r = []
        for a in self.aliases:
            r.append(a.resolve(aliasmap, memo))
        return MultiWrapper(filter(None, r))