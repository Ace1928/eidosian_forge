from zope.interface import Attribute, Interface
class IChatClient(Interface):
    """Interface through which IChatService interacts with clients."""
    name = Attribute('A short string, unique among users.  This will be set by the L{IChatService} at login time.')

    def receive(sender, recipient, message):
        """
        Callback notifying this user of the given message sent by the
        given user.

        This will be invoked whenever another user sends a message to a
        group this user is participating in, or whenever another user sends
        a message directly to this user.  In the former case, C{recipient}
        will be the group to which the message was sent; in the latter, it
        will be the same object as the user who is receiving the message.

        @type sender: L{IUser}
        @type recipient: L{IUser} or L{IGroup}
        @type message: C{dict}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires when the message has been delivered,
        or which fails in some way.  If the Deferred fails and the message
        was directed at a group, this user will be removed from that group.
        """

    def groupMetaUpdate(group, meta):
        """
        Callback notifying this user that the metadata for the given
        group has changed.

        @type group: L{IGroup}
        @type meta: C{dict}

        @rtype: L{twisted.internet.defer.Deferred}
        """

    def userJoined(group, user):
        """
        Callback notifying this user that the given user has joined
        the given group.

        @type group: L{IGroup}
        @type user: L{IUser}

        @rtype: L{twisted.internet.defer.Deferred}
        """

    def userLeft(group, user, reason=None):
        """
        Callback notifying this user that the given user has left the
        given group for the given reason.

        @type group: L{IGroup}
        @type user: L{IUser}
        @type reason: C{unicode}

        @rtype: L{twisted.internet.defer.Deferred}
        """