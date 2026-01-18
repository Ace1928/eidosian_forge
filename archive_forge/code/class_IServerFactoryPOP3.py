from zope.interface import Interface
class IServerFactoryPOP3(Interface):
    """
    An interface for querying capabilities of a POP3 server.

    Any cap_* method may raise L{NotImplementedError} if the particular
    capability is not supported. If L{cap_EXPIRE()} does not raise
    L{NotImplementedError}, L{perUserExpiration()} must be implemented,
    otherwise they are optional. If L{cap_LOGIN_DELAY()} is implemented,
    L{perUserLoginDelay()} must be implemented, otherwise they are optional.

    @type challengers: L{dict} of L{bytes} -> L{IUsernameHashedPassword
        <cred.credentials.IUsernameHashedPassword>}
    @ivar challengers: A mapping of challenger names to
        L{IUsernameHashedPassword <cred.credentials.IUsernameHashedPassword>}
        provider.
    """

    def cap_IMPLEMENTATION():
        """
        Return a string describing the POP3 server implementation.

        @rtype: L{bytes}
        @return: Server implementation information.
        """

    def cap_EXPIRE():
        """
        Return the minimum number of days messages are retained.

        @rtype: L{int} or L{None}
        @return: The minimum number of days messages are retained or none, if
            the server never deletes messages.
        """

    def perUserExpiration():
        """
        Indicate whether the message expiration policy differs per user.

        @rtype: L{bool}
        @return: C{True} when the message expiration policy differs per user,
            C{False} otherwise.
        """

    def cap_LOGIN_DELAY():
        """
        Return the minimum number of seconds between client logins.

        @rtype: L{int}
        @return: The minimum number of seconds between client logins.
        """

    def perUserLoginDelay():
        """
        Indicate whether the login delay period differs per user.

        @rtype: L{bool}
        @return: C{True} when the login delay differs per user, C{False}
            otherwise.
        """