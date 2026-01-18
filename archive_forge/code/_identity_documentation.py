import abc
from ._error import IdentityError
Parses the identity declaration from the given declared attributes.

        TODO take the set of first party caveat conditions instead?
        @param ctx (AuthContext)
        @param declared (dict of string/string)
        :return: an Identity
        