from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
@implementer(portal.IRealm, iwords.IChatService)
class WordsRealm:
    _encoding = 'utf-8'

    def __init__(self, name):
        self.name = name

    def userFactory(self, name):
        return User(name)

    def groupFactory(self, name):
        return Group(name)

    def logoutFactory(self, avatar, facet):

        def logout():
            getattr(facet, 'logout', lambda: None)()
            avatar.realm = avatar.mind = None
        return logout

    def requestAvatar(self, avatarId, mind, *interfaces):
        if isinstance(avatarId, bytes):
            avatarId = avatarId.decode(self._encoding)

        def gotAvatar(avatar):
            if avatar.realm is not None:
                raise ewords.AlreadyLoggedIn()
            for iface in interfaces:
                facet = iface(avatar, None)
                if facet is not None:
                    avatar.loggedIn(self, mind)
                    mind.name = avatarId
                    mind.realm = self
                    mind.avatar = avatar
                    return (iface, facet, self.logoutFactory(avatar, facet))
            raise NotImplementedError(self, interfaces)
        return self.getUser(avatarId).addCallback(gotAvatar)

    def itergroups(self):
        pass
    createGroupOnRequest = False
    createUserOnRequest = True

    def lookupUser(self, name):
        raise NotImplementedError

    def lookupGroup(self, group):
        raise NotImplementedError

    def addUser(self, user):
        """
        Add the given user to this service.

        This is an internal method intended to be overridden by
        L{WordsRealm} subclasses, not called by external code.

        @type user: L{IUser}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with L{None} when the user is
        added, or which fails with
        L{twisted.words.ewords.DuplicateUser} if a user with the
        same name exists already.
        """
        raise NotImplementedError

    def addGroup(self, group):
        """
        Add the given group to this service.

        @type group: L{IGroup}

        @rtype: L{twisted.internet.defer.Deferred}
        @return: A Deferred which fires with L{None} when the group is
        added, or which fails with
        L{twisted.words.ewords.DuplicateGroup} if a group with the
        same name exists already.
        """
        raise NotImplementedError

    def getGroup(self, name):
        if self.createGroupOnRequest:

            def ebGroup(err):
                err.trap(ewords.DuplicateGroup)
                return self.lookupGroup(name)
            return self.createGroup(name).addErrback(ebGroup)
        return self.lookupGroup(name)

    def getUser(self, name):
        if self.createUserOnRequest:

            def ebUser(err):
                err.trap(ewords.DuplicateUser)
                return self.lookupUser(name)
            return self.createUser(name).addErrback(ebUser)
        return self.lookupUser(name)

    def createUser(self, name):

        def cbLookup(user):
            return failure.Failure(ewords.DuplicateUser(name))

        def ebLookup(err):
            err.trap(ewords.NoSuchUser)
            return self.userFactory(name)
        name = name.lower()
        d = self.lookupUser(name)
        d.addCallbacks(cbLookup, ebLookup)
        d.addCallback(self.addUser)
        return d

    def createGroup(self, name):

        def cbLookup(group):
            return failure.Failure(ewords.DuplicateGroup(name))

        def ebLookup(err):
            err.trap(ewords.NoSuchGroup)
            return self.groupFactory(name)
        name = name.lower()
        d = self.lookupGroup(name)
        d.addCallbacks(cbLookup, ebLookup)
        d.addCallback(self.addGroup)
        return d