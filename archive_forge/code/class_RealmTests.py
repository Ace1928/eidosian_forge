import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
class RealmTests(unittest.TestCase):

    def _entityCreationTest(self, kind):
        realm = service.InMemoryWordsRealm('realmname')
        name = 'test' + kind.lower()
        create = getattr(realm, 'create' + kind.title())
        get = getattr(realm, 'get' + kind.title())
        flag = 'create' + kind.title() + 'OnRequest'
        dupExc = getattr(ewords, 'Duplicate' + kind.title())
        noSuchExc = getattr(ewords, 'NoSuch' + kind.title())
        p = self.successResultOf(create(name))
        self.assertEqual(name, p.name)
        self.failureResultOf(create(name)).trap(dupExc)
        setattr(realm, flag, True)
        p = self.successResultOf(get('new' + kind.lower()))
        self.assertEqual('new' + kind.lower(), p.name)
        newp = self.successResultOf(get('new' + kind.lower()))
        self.assertIdentical(p, newp)
        setattr(realm, flag, False)
        self.failureResultOf(get('another' + kind.lower())).trap(noSuchExc)

    def testUserCreation(self):
        return self._entityCreationTest('User')

    def testGroupCreation(self):
        return self._entityCreationTest('Group')

    def testUserRetrieval(self):
        realm = service.InMemoryWordsRealm('realmname')
        user = self.successResultOf(realm.createUser('testuser'))
        retrieved = self.successResultOf(realm.getUser('testuser'))
        self.assertIdentical(user, retrieved)
        lookedUp = self.successResultOf(realm.lookupUser('testuser'))
        self.assertIdentical(retrieved, lookedUp)
        self.failureResultOf(realm.lookupUser('nosuchuser')).trap(ewords.NoSuchUser)

    def testUserAddition(self):
        realm = service.InMemoryWordsRealm('realmname')
        p = service.User('testuser')
        user = self.successResultOf(realm.addUser(p))
        self.assertIdentical(p, user)
        retrieved = self.successResultOf(realm.getUser('testuser'))
        self.assertIdentical(user, retrieved)
        lookedUp = self.successResultOf(realm.lookupUser('testuser'))
        self.assertIdentical(retrieved, lookedUp)

    def testGroupRetrieval(self):
        realm = service.InMemoryWordsRealm('realmname')
        group = self.successResultOf(realm.createGroup('testgroup'))
        retrieved = self.successResultOf(realm.getGroup('testgroup'))
        self.assertIdentical(group, retrieved)
        self.failureResultOf(realm.getGroup('nosuchgroup')).trap(ewords.NoSuchGroup)

    def testGroupAddition(self):
        realm = service.InMemoryWordsRealm('realmname')
        p = service.Group('testgroup')
        self.successResultOf(realm.addGroup(p))
        group = self.successResultOf(realm.getGroup('testGroup'))
        self.assertIdentical(p, group)

    def testGroupUsernameCollision(self):
        """
        Try creating a group with the same name as an existing user and
        assert that it succeeds, since users and groups should not be in the
        same namespace and collisions should be impossible.
        """
        realm = service.InMemoryWordsRealm('realmname')
        self.successResultOf(realm.createUser('test'))
        self.successResultOf(realm.createGroup('test'))

    def testEnumeration(self):
        realm = service.InMemoryWordsRealm('realmname')
        self.successResultOf(realm.createGroup('groupone'))
        self.successResultOf(realm.createGroup('grouptwo'))
        groups = self.successResultOf(realm.itergroups())
        n = [g.name for g in groups]
        n.sort()
        self.assertEqual(n, ['groupone', 'grouptwo'])