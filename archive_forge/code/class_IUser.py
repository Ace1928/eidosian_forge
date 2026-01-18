from zope.interface import Attribute, Interface
class IUser(Interface):
    """Interface through which clients interact with IChatService."""
    realm = Attribute('A reference to the Realm to which this user belongs.  Set if and only if the user is logged in.')
    mind = Attribute('A reference to the mind which logged in to this user.  Set if and only if the user is logged in.')
    name = Attribute('A short string, unique among users.')
    lastMessage = Attribute('A POSIX timestamp indicating the time of the last message received from this user.')
    signOn = Attribute("A POSIX timestamp indicating this user's most recent sign on time.")

    def loggedIn(realm, mind):
        """Invoked by the associated L{IChatService} when login occurs.

        @param realm: The L{IChatService} through which login is occurring.
        @param mind: The mind object used for cred login.
        """

    def send(recipient, message):
        """Send the given message to the given user or group.

        @type recipient: Either L{IUser} or L{IGroup}
        @type message: C{dict}
        """

    def join(group):
        """Attempt to join the given group.

        @type group: L{IGroup}
        @rtype: L{twisted.internet.defer.Deferred}
        """

    def leave(group):
        """Discontinue participation in the given group.

        @type group: L{IGroup}
        @rtype: L{twisted.internet.defer.Deferred}
        """

    def itergroups():
        """
        Return an iterator of all groups of which this user is a
        member.
        """