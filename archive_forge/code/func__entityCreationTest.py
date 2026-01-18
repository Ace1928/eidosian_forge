import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
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