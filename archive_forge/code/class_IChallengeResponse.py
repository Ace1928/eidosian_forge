from zope.interface import Interface
class IChallengeResponse(Interface):
    """
    An C{IMAPrev4} authorization challenge mechanism.
    """

    def getChallenge():
        """
        Return a client challenge.

        @return: A challenge.
        @rtype: L{bytes}
        """

    def setResponse(response):
        """
        Extract a username and possibly a password from a response and
        assign them to C{username} and C{password} instance variables.

        @param response: A decoded response.
        @type response: L{bytes}

        @see: L{credentials.IUsernamePassword} or
            L{credentials.IUsernameHashedPassword}
        """

    def moreChallenges():
        """
        Are there more challenges than just the first?  If so, callers
        should challenge clients with the result of L{getChallenge},
        and check their response with L{setResponse} in a loop until
        this returns L{False}

        @return: Are there more challenges?
        @rtype: L{bool}
        """