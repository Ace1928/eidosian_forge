from zope.interface import Attribute, Interface
class ISession(Interface):

    def getPty(term, windowSize, modes):
        """
        Get a pseudo-terminal for use by a shell or command.

        If a pseudo-terminal is not available, or the request otherwise
        fails, raise an exception.
        """

    def openShell(proto):
        """
        Open a shell and connect it to proto.

        @param proto: a L{ProcessProtocol} instance.
        """

    def execCommand(proto, command):
        """
        Execute a command.

        @param proto: a L{ProcessProtocol} instance.
        """

    def windowChanged(newWindowSize):
        """
        Called when the size of the remote screen has changed.
        """

    def eofReceived():
        """
        Called when the other side has indicated no more data will be sent.
        """

    def closed():
        """
        Called when the session is closed.
        """