from typing import Dict, Tuple, Union
from twisted.words.protocols.jabber.xmpp_stringprep import (
class JID:
    """
    Represents a stringprep'd Jabber ID.

    JID objects are hashable so they can be used in sets and as keys in
    dictionaries.
    """

    def __init__(self, str: Union[str, None]=None, tuple: Union[Tuple[Union[str, None], str, Union[str, None]], None]=None):
        if str:
            user, host, res = parse(str)
        elif tuple:
            user, host, res = prep(*tuple)
        else:
            raise RuntimeError("You must provide a value for either 'str' or 'tuple' arguments.")
        self.user = user
        self.host = host
        self.resource = res

    def userhost(self):
        """
        Extract the bare JID as a unicode string.

        A bare JID does not have a resource part, so this returns either
        C{user@host} or just C{host}.

        @rtype: L{str}
        """
        if self.user:
            return f'{self.user}@{self.host}'
        else:
            return self.host

    def userhostJID(self):
        """
        Extract the bare JID.

        A bare JID does not have a resource part, so this returns a
        L{JID} object representing either C{user@host} or just C{host}.

        If the object this method is called upon doesn't have a resource
        set, it will return itself. Otherwise, the bare JID object will
        be created, interned using L{internJID}.

        @rtype: L{JID}
        """
        if self.resource:
            return internJID(self.userhost())
        else:
            return self

    def full(self):
        """
        Return the string representation of this JID.

        @rtype: L{str}
        """
        if self.user:
            if self.resource:
                return f'{self.user}@{self.host}/{self.resource}'
            else:
                return f'{self.user}@{self.host}'
        elif self.resource:
            return f'{self.host}/{self.resource}'
        else:
            return self.host

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison.

        L{JID}s compare equal if their user, host and resource parts all
        compare equal.  When comparing against instances of other types, it
        uses the default comparison.
        """
        if isinstance(other, JID):
            return self.user == other.user and self.host == other.host and (self.resource == other.resource)
        else:
            return NotImplemented

    def __hash__(self):
        """
        Calculate hash.

        L{JID}s with identical constituent user, host and resource parts have
        equal hash values.  In combination with the comparison defined on JIDs,
        this allows for using L{JID}s in sets and as dictionary keys.
        """
        return hash((self.user, self.host, self.resource))

    def __unicode__(self):
        """
        Get unicode representation.

        Return the string representation of this JID as a unicode string.
        @see: L{full}
        """
        return self.full()
    __str__ = __unicode__

    def __repr__(self) -> str:
        """
        Get object representation.

        Returns a string that would create a new JID object that compares equal
        to this one.
        """
        return 'JID(%r)' % self.full()