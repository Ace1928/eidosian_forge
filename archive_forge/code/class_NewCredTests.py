import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
class NewCredTests(unittest.TestCase):
    """
    Tests related to the L{twisted.cred} support in PB.
    """

    def setUp(self):
        """
        Create a portal with no checkers and wrap it around a simple test
        realm.  Set up a PB server on a TCP port which serves perspectives
        using that portal.
        """
        self.realm = TestRealm()
        self.portal = portal.Portal(self.realm)
        self.serverFactory = ConnectionNotifyServerFactory(self.portal)
        self.clientFactory = pb.PBClientFactory()

    def establishClientAndServer(self, _ignored=None):
        """
        Connect a client obtained from C{clientFactory} and a server
        obtained from the current server factory via an L{IOPump},
        then assign them to the appropriate instance variables

        @ivar clientFactory: the broker client factory
        @ivar clientFactory: L{pb.PBClientFactory} instance

        @ivar client: the client broker
        @type client: L{pb.Broker}

        @ivar server: the server broker
        @type server: L{pb.Broker}

        @ivar pump: the IOPump connecting the client and server
        @type pump: L{IOPump}

        @ivar connector: A connector whose connect method recreates
            the above instance variables
        @type connector: L{twisted.internet.base.IConnector}
        """
        self.client, self.server, self.pump = connectServerAndClient(self, self.clientFactory, self.serverFactory)
        self.connectorState = _ReconnectingFakeConnectorState()
        self.connector = _ReconnectingFakeConnector(address.IPv4Address('TCP', '127.0.0.1', 4321), self.connectorState)
        self.connectorState.notifyOnConnect().addCallback(self.establishClientAndServer)

    def completeClientLostConnection(self, reason=failure.Failure(main.CONNECTION_DONE)):
        """
        Asserts that the client broker's transport was closed and then
        mimics the event loop by calling the broker's connectionLost
        callback with C{reason}, followed by C{self.clientFactory}'s
        C{clientConnectionLost}

        @param reason: (optional) the reason to pass to the client
            broker's connectionLost callback
        @type reason: L{Failure}
        """
        self.assertTrue(self.client.transport.closed)
        self.client.connectionLost(reason)
        self.clientFactory.clientConnectionLost(self.connector, reason)

    def test_getRootObject(self):
        """
        Assert that L{PBClientFactory.getRootObject}'s Deferred fires with
        a L{RemoteReference}, and that disconnecting it runs its
        disconnection callbacks.
        """
        self.establishClientAndServer()
        rootObjDeferred = self.clientFactory.getRootObject()

        def gotRootObject(rootObj):
            self.assertIsInstance(rootObj, pb.RemoteReference)
            return rootObj

        def disconnect(rootObj):
            disconnectedDeferred = Deferred()
            rootObj.notifyOnDisconnect(disconnectedDeferred.callback)
            self.clientFactory.disconnect()
            self.completeClientLostConnection()
            return disconnectedDeferred
        rootObjDeferred.addCallback(gotRootObject)
        rootObjDeferred.addCallback(disconnect)
        return rootObjDeferred

    def test_deadReferenceError(self):
        """
        Test that when a connection is lost, calling a method on a
        RemoteReference obtained from it raises L{DeadReferenceError}.
        """
        self.establishClientAndServer()
        rootObjDeferred = self.clientFactory.getRootObject()

        def gotRootObject(rootObj):
            disconnectedDeferred = Deferred()
            rootObj.notifyOnDisconnect(disconnectedDeferred.callback)

            def lostConnection(ign):
                self.assertRaises(pb.DeadReferenceError, rootObj.callRemote, 'method')
            disconnectedDeferred.addCallback(lostConnection)
            self.clientFactory.disconnect()
            self.completeClientLostConnection()
            return disconnectedDeferred
        return rootObjDeferred.addCallback(gotRootObject)

    def test_clientConnectionLost(self):
        """
        Test that if the L{reconnecting} flag is passed with a True value then
        a remote call made from a disconnection notification callback gets a
        result successfully.
        """

        class ReconnectOnce(pb.PBClientFactory):
            reconnectedAlready = False

            def clientConnectionLost(self, connector, reason):
                reconnecting = not self.reconnectedAlready
                self.reconnectedAlready = True
                result = pb.PBClientFactory.clientConnectionLost(self, connector, reason, reconnecting)
                if reconnecting:
                    connector.connect()
                return result
        self.clientFactory = ReconnectOnce()
        self.establishClientAndServer()
        rootObjDeferred = self.clientFactory.getRootObject()

        def gotRootObject(rootObj):
            self.assertIsInstance(rootObj, pb.RemoteReference)
            d = Deferred()
            rootObj.notifyOnDisconnect(d.callback)
            self.clientFactory.disconnect()
            self.completeClientLostConnection()

            def disconnected(ign):
                d = self.clientFactory.getRootObject()

                def gotAnotherRootObject(anotherRootObj):
                    self.assertIsInstance(anotherRootObj, pb.RemoteReference)
                    d = Deferred()
                    anotherRootObj.notifyOnDisconnect(d.callback)
                    self.clientFactory.disconnect()
                    self.completeClientLostConnection()
                    return d
                return d.addCallback(gotAnotherRootObject)
            return d.addCallback(disconnected)
        return rootObjDeferred.addCallback(gotRootObject)

    def test_immediateClose(self):
        """
        Test that if a Broker loses its connection without receiving any bytes,
        it doesn't raise any exceptions or log any errors.
        """
        self.establishClientAndServer()
        serverProto = self.serverFactory.buildProtocol(('127.0.0.1', 12345))
        serverProto.makeConnection(protocol.FileWrapper(StringIO()))
        serverProto.connectionLost(failure.Failure(main.CONNECTION_DONE))

    def test_loginConnectionRefused(self):
        """
        L{PBClientFactory.login} returns a L{Deferred} which is errbacked
        with the L{ConnectionRefusedError} if the underlying connection is
        refused.
        """
        clientFactory = pb.PBClientFactory()
        loginDeferred = clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'))
        clientFactory.clientConnectionFailed(None, failure.Failure(ConnectionRefusedError('Test simulated refused connection')))
        return self.assertFailure(loginDeferred, ConnectionRefusedError)

    def test_loginLogout(self):
        """
        Test that login can be performed with IUsernamePassword credentials and
        that when the connection is dropped the avatar is logged out.
        """
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        creds = credentials.UsernamePassword(b'user', b'pass')
        mind = 'BRAINS!'
        loginCompleted = Deferred()
        d = self.clientFactory.login(creds, mind)

        def cbLogin(perspective):
            self.assertTrue(self.realm.lastPerspective.loggedIn)
            self.assertIsInstance(perspective, pb.RemoteReference)
            return loginCompleted

        def cbDisconnect(ignored):
            self.clientFactory.disconnect()
            self.completeClientLostConnection()
        d.addCallback(cbLogin)
        d.addCallback(cbDisconnect)

        def cbLogout(ignored):
            self.assertTrue(self.realm.lastPerspective.loggedOut)
        d.addCallback(cbLogout)
        self.establishClientAndServer()
        self.pump.flush()
        gc.collect()
        self.pump.flush()
        loginCompleted.callback(None)
        return d

    def test_logoutAfterDecref(self):
        """
        If a L{RemoteReference} to an L{IPerspective} avatar is decrefed and
        there remain no other references to the avatar on the server, the
        avatar is garbage collected and the logout method called.
        """
        loggedOut = Deferred()

        class EventPerspective(pb.Avatar):
            """
            An avatar which fires a Deferred when it is logged out.
            """

            def __init__(self, avatarId):
                pass

            def logout(self):
                loggedOut.callback(None)
        self.realm.perspectiveFactory = EventPerspective
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(foo=b'bar'))
        d = self.clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'), 'BRAINS!')

        def cbLoggedIn(avatar):
            return loggedOut
        d.addCallback(cbLoggedIn)

        def cbLoggedOut(ignored):
            self.assertEqual(self.serverFactory.protocolInstance._localCleanup, {})
        d.addCallback(cbLoggedOut)
        self.establishClientAndServer()
        self.pump.flush()
        gc.collect()
        self.pump.flush()
        return d

    def test_concurrentLogin(self):
        """
        Two different correct login attempts can be made on the same root
        object at the same time and produce two different resulting avatars.
        """
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(foo=b'bar', baz=b'quux'))
        firstLogin = self.clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'), 'BRAINS!')
        secondLogin = self.clientFactory.login(credentials.UsernamePassword(b'baz', b'quux'), 'BRAINS!')
        d = gatherResults([firstLogin, secondLogin])

        def cbLoggedIn(result):
            first, second = result
            return gatherResults([first.callRemote('getAvatarId'), second.callRemote('getAvatarId')])
        d.addCallback(cbLoggedIn)

        def cbAvatarIds(x):
            first, second = x
            self.assertEqual(first, b'foo')
            self.assertEqual(second, b'baz')
        d.addCallback(cbAvatarIds)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_badUsernamePasswordLogin(self):
        """
        Test that a login attempt with an invalid user or invalid password
        fails in the appropriate way.
        """
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        firstLogin = self.clientFactory.login(credentials.UsernamePassword(b'nosuchuser', b'pass'))
        secondLogin = self.clientFactory.login(credentials.UsernamePassword(b'user', b'wrongpass'))
        self.assertFailure(firstLogin, UnauthorizedLogin)
        self.assertFailure(secondLogin, UnauthorizedLogin)
        d = gatherResults([firstLogin, secondLogin])

        def cleanup(ignore):
            errors = self.flushLoggedErrors(UnauthorizedLogin)
            self.assertEqual(len(errors), 2)
        d.addCallback(cleanup)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_anonymousLogin(self):
        """
        Verify that a PB server using a portal configured with a checker which
        allows IAnonymous credentials can be logged into using IAnonymous
        credentials.
        """
        self.portal.registerChecker(checkers.AllowAnonymousAccess())
        d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')

        def cbLoggedIn(perspective):
            return perspective.callRemote('echo', 123)
        d.addCallback(cbLoggedIn)
        d.addCallback(self.assertEqual, 123)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_anonymousLoginNotPermitted(self):
        """
        Verify that without an anonymous checker set up, anonymous login is
        rejected.
        """
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user='pass'))
        d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')
        self.assertFailure(d, UnhandledCredentials)

        def cleanup(ignore):
            errors = self.flushLoggedErrors(UnhandledCredentials)
            self.assertEqual(len(errors), 1)
        d.addCallback(cleanup)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_anonymousLoginWithMultipleCheckers(self):
        """
        Like L{test_anonymousLogin} but against a portal with a checker for
        both IAnonymous and IUsernamePassword.
        """
        self.portal.registerChecker(checkers.AllowAnonymousAccess())
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')

        def cbLogin(perspective):
            return perspective.callRemote('echo', 123)
        d.addCallback(cbLogin)
        d.addCallback(self.assertEqual, 123)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_authenticatedLoginWithMultipleCheckers(self):
        """
        Like L{test_anonymousLoginWithMultipleCheckers} but check that
        username/password authentication works.
        """
        self.portal.registerChecker(checkers.AllowAnonymousAccess())
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        d = self.clientFactory.login(credentials.UsernamePassword(b'user', b'pass'), 'BRAINS!')

        def cbLogin(perspective):
            return perspective.callRemote('add', 100, 23)
        d.addCallback(cbLogin)
        d.addCallback(self.assertEqual, 123)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_view(self):
        """
        Verify that a viewpoint can be retrieved after authenticating with
        cred.
        """
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        d = self.clientFactory.login(credentials.UsernamePassword(b'user', b'pass'), 'BRAINS!')

        def cbLogin(perspective):
            return perspective.callRemote('getViewPoint')
        d.addCallback(cbLogin)

        def cbView(viewpoint):
            return viewpoint.callRemote('check')
        d.addCallback(cbView)
        d.addCallback(self.assertTrue)
        self.establishClientAndServer()
        self.pump.flush()
        return d