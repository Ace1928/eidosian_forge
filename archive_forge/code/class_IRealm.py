from typing import Callable, Dict, Iterable, List, Tuple, Type, Union
from zope.interface import Interface, providedBy
from twisted.cred import error
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import ICredentials
from twisted.internet import defer
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.python import failure, reflect
class IRealm(Interface):
    """
    The realm connects application-specific objects to the
    authentication system.
    """

    def requestAvatar(avatarId: Union[bytes, Tuple[()]], mind: object, *interfaces: _InterfaceItself) -> Union[Deferred[_requestResult], _requestResult]:
        """
        Return avatar which provides one of the given interfaces.

        @param avatarId: a string that identifies an avatar, as returned by
            L{ICredentialsChecker.requestAvatarId<twisted.cred.checkers.ICredentialsChecker.requestAvatarId>}
            (via a Deferred).  Alternatively, it may be
            C{twisted.cred.checkers.ANONYMOUS}.
        @param mind: usually None.  See the description of mind in
            L{Portal.login}.
        @param interfaces: the interface(s) the returned avatar should
            implement, e.g.  C{IMailAccount}.  See the description of
            L{Portal.login}.

        @returns: a deferred which will fire a tuple of (interface,
            avatarAspect, logout), or the tuple itself.  The interface will be
            one of the interfaces passed in the 'interfaces' argument.  The
            'avatarAspect' will implement that interface.  The 'logout' object
            is a callable which will detach the mind from the avatar.
        """