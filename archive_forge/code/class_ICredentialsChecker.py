import os
from typing import Any, Dict, Optional, Tuple, Union
from zope.interface import Attribute, Interface, implementer
from twisted.cred import error
from twisted.cred.credentials import (
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import failure
class ICredentialsChecker(Interface):
    """
    An object that can check sub-interfaces of L{ICredentials}.
    """
    credentialInterfaces = Attribute('A list of sub-interfaces of L{ICredentials} which specifies which I may check.')

    def requestAvatarId(credentials: Any) -> Deferred[Union[bytes, Tuple[()]]]:
        """
        Validate credentials and produce an avatar ID.

        @param credentials: something which implements one of the interfaces in
            C{credentialInterfaces}.

        @return: a L{Deferred} which will fire with a L{bytes} that identifies
            an avatar, an empty tuple to specify an authenticated anonymous
            user (provided as L{twisted.cred.checkers.ANONYMOUS}) or fail with
            L{UnauthorizedLogin}.  Alternatively, return the result itself.

        @see: L{twisted.cred.credentials}
        """