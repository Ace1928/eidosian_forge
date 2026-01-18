from zope.interface import Attribute, Interface
class IConchUser(Interface):
    """
    A user who has been authenticated to Cred through Conch.  This is
    the interface between the SSH connection and the user.
    """
    conn = Attribute('The SSHConnection object for this user.')

    def lookupChannel(channelType, windowSize, maxPacket, data):
        """
        The other side requested a channel of some sort.

        C{channelType} is the type of channel being requested,
        as an ssh connection protocol channel type.
        C{data} is any other packet data (often nothing).

        We return a subclass of L{SSHChannel<ssh.channel.SSHChannel>}.  If
        the channel type is unknown, we return C{None}.

        For other failures, we raise an exception. If a
        L{ConchError<error.ConchError>} is raised, the C{.value} will
        be the message, and the C{.data} will be the error code.

        @param channelType: The requested channel type
        @type channelType:  L{bytes}
        @param windowSize:  The initial size of the remote window
        @type windowSize:   L{int}
        @param maxPacket:   The largest packet we should send
        @type maxPacket:    L{int}
        @param data:        Additional request data
        @type data:         L{bytes}
        @rtype:             a subclass of L{SSHChannel} or L{None}
        """

    def lookupSubsystem(subsystem, data):
        """
        The other side requested a subsystem.

        We return a L{Protocol} implementing the requested subsystem.
        If the subsystem is not available, we return C{None}.

        @param subsystem: The name of the subsystem being requested
        @type subsystem: L{bytes}
        @param data:     Additional request data (often nothing)
        @type data:      L{bytes}
        @rtype:          L{Protocol} or L{None}
        """

    def gotGlobalRequest(requestType, data):
        """
        A global request was sent from the other side.

        We return a true value on success or a false value on failure.
        If we indicate success by returning a tuple, its second item
        will be sent to the other side as additional response data.

        @param requestType: The type of the request
        @type requestType:  L{bytes}
        @param data:        Additional request data
        @type data:         L{bytes}
        @rtype:             boolean or L{tuple}
        """