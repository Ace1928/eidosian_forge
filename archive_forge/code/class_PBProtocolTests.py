import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
class PBProtocolTests(unittest.TestCase):

    def setUp(self):
        self.realm = service.InMemoryWordsRealm('realmname')
        self.checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.portal = portal.Portal(self.realm, [self.checker])
        self.serverFactory = pb.PBServerFactory(self.portal)
        self.serverFactory.protocol = self._protocolFactory
        self.serverFactory.unsafeTracebacks = True
        self.clientFactory = pb.PBClientFactory()
        self.clientFactory.unsafeTracebacks = True
        self.serverPort = reactor.listenTCP(0, self.serverFactory)
        self.clientConn = reactor.connectTCP('127.0.0.1', self.serverPort.getHost().port, self.clientFactory)

    def _protocolFactory(self, *args, **kw):
        self._serverProtocol = pb.Broker(0)
        return self._serverProtocol

    def tearDown(self):
        d3 = Deferred()
        self._serverProtocol.notifyOnDisconnect(lambda: d3.callback(None))
        return DeferredList([maybeDeferred(self.serverPort.stopListening), maybeDeferred(self.clientConn.disconnect), d3])

    def _loggedInAvatar(self, name, password, mind):
        nameBytes = name
        if isinstance(name, str):
            nameBytes = name.encode('ascii')
        creds = credentials.UsernamePassword(nameBytes, password)
        self.checker.addUser(nameBytes, password)
        d = self.realm.createUser(name)
        d.addCallback(lambda ign: self.clientFactory.login(creds, mind))
        return d

    @defer.inlineCallbacks
    def testGroups(self):
        mindone = TestMind()
        one = (yield self._loggedInAvatar('one', b'p1', mindone))
        mindtwo = TestMind()
        two = (yield self._loggedInAvatar('two', b'p2', mindtwo))
        mindThree = TestMind()
        three = (yield self._loggedInAvatar(b'three', b'p3', mindThree))
        yield self.realm.createGroup('foobar')
        yield self.realm.createGroup(b'barfoo')
        groupone = (yield one.join('foobar'))
        grouptwo = (yield two.join(b'barfoo'))
        yield two.join('foobar')
        yield two.join(b'barfoo')
        yield three.join('foobar')
        yield groupone.send({b'text': b'hello, monkeys'})
        yield groupone.leave()
        yield grouptwo.leave()