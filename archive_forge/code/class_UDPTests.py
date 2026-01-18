import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
@skipIf(not interfaces.IReactorUDP(reactor, None), 'This reactor does not support UDP')
class UDPTests(TestCase):

    def test_oldAddress(self):
        """
        The C{type} of the host address of a listening L{DatagramProtocol}'s
        transport is C{"UDP"}.
        """
        server = Server()
        d = server.startedDeferred = defer.Deferred()
        p = reactor.listenUDP(0, server, interface='127.0.0.1')

        def cbStarted(ignored):
            addr = p.getHost()
            self.assertEqual(addr.type, 'UDP')
            return p.stopListening()
        return d.addCallback(cbStarted)

    def test_startStop(self):
        """
        The L{DatagramProtocol}'s C{startProtocol} and C{stopProtocol}
        methods are called when its transports starts and stops listening,
        respectively.
        """
        server = Server()
        d = server.startedDeferred = defer.Deferred()
        port1 = reactor.listenUDP(0, server, interface='127.0.0.1')

        def cbStarted(ignored):
            self.assertEqual(server.started, 1)
            self.assertEqual(server.stopped, 0)
            return port1.stopListening()

        def cbStopped(ignored):
            self.assertEqual(server.stopped, 1)
        return d.addCallback(cbStarted).addCallback(cbStopped)

    def test_rebind(self):
        """
        Re-listening with the same L{DatagramProtocol} re-invokes the
        C{startProtocol} callback.
        """
        server = Server()
        d = server.startedDeferred = defer.Deferred()
        p = reactor.listenUDP(0, server, interface='127.0.0.1')

        def cbStarted(ignored, port):
            return port.stopListening()

        def cbStopped(ignored):
            d = server.startedDeferred = defer.Deferred()
            p = reactor.listenUDP(0, server, interface='127.0.0.1')
            return d.addCallback(cbStarted, p)
        return d.addCallback(cbStarted, p)

    def test_bindError(self):
        """
        A L{CannotListenError} exception is raised when attempting to bind a
        second protocol instance to an already bound port
        """
        server = Server()
        d = server.startedDeferred = defer.Deferred()
        port = reactor.listenUDP(0, server, interface='127.0.0.1')

        def cbStarted(ignored):
            self.assertEqual(port.getHost(), server.transport.getHost())
            server2 = Server()
            self.assertRaises(error.CannotListenError, reactor.listenUDP, port.getHost().port, server2, interface='127.0.0.1')
        d.addCallback(cbStarted)

        def cbFinished(ignored):
            return port.stopListening()
        d.addCallback(cbFinished)
        return d

    def test_sendPackets(self):
        """
        Datagrams can be sent with the transport's C{write} method and
        received via the C{datagramReceived} callback method.
        """
        server = Server()
        serverStarted = server.startedDeferred = defer.Deferred()
        port1 = reactor.listenUDP(0, server, interface='127.0.0.1')
        client = GoodClient()
        clientStarted = client.startedDeferred = defer.Deferred()

        def cbServerStarted(ignored):
            self.port2 = reactor.listenUDP(0, client, interface='127.0.0.1')
            return clientStarted
        d = serverStarted.addCallback(cbServerStarted)

        def cbClientStarted(ignored):
            client.transport.connect('127.0.0.1', server.transport.getHost().port)
            cAddr = client.transport.getHost()
            sAddr = server.transport.getHost()
            serverSend = client.packetReceived = defer.Deferred()
            server.transport.write(b'hello', (cAddr.host, cAddr.port))
            clientWrites = [(b'a',), (b'b', None), (b'c', (sAddr.host, sAddr.port))]

            def cbClientSend(ignored):
                if clientWrites:
                    nextClientWrite = server.packetReceived = defer.Deferred()
                    nextClientWrite.addCallback(cbClientSend)
                    client.transport.write(*clientWrites.pop(0))
                    return nextClientWrite
            return defer.DeferredList([cbClientSend(None), serverSend], fireOnOneErrback=True)
        d.addCallback(cbClientStarted)

        def cbSendsFinished(ignored):
            cAddr = client.transport.getHost()
            sAddr = server.transport.getHost()
            self.assertEqual(client.packets, [(b'hello', (sAddr.host, sAddr.port))])
            clientAddr = (cAddr.host, cAddr.port)
            self.assertEqual(server.packets, [(b'a', clientAddr), (b'b', clientAddr), (b'c', clientAddr)])
        d.addCallback(cbSendsFinished)

        def cbFinished(ignored):
            return defer.DeferredList([defer.maybeDeferred(port1.stopListening), defer.maybeDeferred(self.port2.stopListening)], fireOnOneErrback=True)
        d.addCallback(cbFinished)
        return d

    @skipIf(os.environ.get('INFRASTRUCTURE') == 'AZUREPIPELINES', 'Hangs on Pipelines due to firewall')
    def test_connectionRefused(self):
        """
        A L{ConnectionRefusedError} exception is raised when a connection
        attempt is actively refused by the other end.

        Note: This test assumes no one is listening on port 80 UDP.
        """
        client = GoodClient()
        clientStarted = client.startedDeferred = defer.Deferred()
        port = reactor.listenUDP(0, client, interface='127.0.0.1')
        server = Server()
        serverStarted = server.startedDeferred = defer.Deferred()
        port2 = reactor.listenUDP(0, server, interface='127.0.0.1')
        d = defer.DeferredList([clientStarted, serverStarted], fireOnOneErrback=True)

        def cbStarted(ignored):
            connectionRefused = client.startedDeferred = defer.Deferred()
            client.transport.connect('127.0.0.1', 80)
            for i in range(10):
                client.transport.write(b'%d' % (i,))
                server.transport.write(b'%d' % (i,), ('127.0.0.1', 80))
            return self.assertFailure(connectionRefused, error.ConnectionRefusedError)
        d.addCallback(cbStarted)

        def cbFinished(ignored):
            return defer.DeferredList([defer.maybeDeferred(port.stopListening), defer.maybeDeferred(port2.stopListening)], fireOnOneErrback=True)
        d.addCallback(cbFinished)
        return d

    def test_serverReadFailure(self):
        """
        When a server fails to successfully read a packet the server should
        still be able to process future packets.
        The IOCP reactor had a historical problem where a failure to read caused
        the reactor to ignore any future reads. This test should prevent a regression.

        Note: This test assumes no one is listening on port 80 UDP.
        """
        client = GoodClient()
        clientStarted = client.startedDeferred = defer.Deferred()
        clientPort = reactor.listenUDP(0, client, interface='127.0.0.1')
        test_data_to_send = b'Sending test packet to server'
        server = Server()
        serverStarted = server.startedDeferred = defer.Deferred()
        serverGotData = server.packetReceived = defer.Deferred()
        serverPort = reactor.listenUDP(0, server, interface='127.0.0.1')
        server_client_started_d = defer.DeferredList([clientStarted, serverStarted], fireOnOneErrback=True)

        def cbClientAndServerStarted(ignored):
            server.transport.write(b'write to port no one is listening to', ('127.0.0.1', 80))
            client.transport.write(test_data_to_send, ('127.0.0.1', serverPort._realPortNumber))
        server_client_started_d.addCallback(cbClientAndServerStarted)
        all_data_sent = defer.DeferredList([server_client_started_d, serverGotData], fireOnOneErrback=True)

        def verify_server_got_data(ignored):
            self.assertEqual(server.packets[0][0], test_data_to_send)
        all_data_sent.addCallback(verify_server_got_data)

        def cleanup(ignored):
            return defer.DeferredList([defer.maybeDeferred(clientPort.stopListening), defer.maybeDeferred(serverPort.stopListening)], fireOnOneErrback=True)
        all_data_sent.addCallback(cleanup)
        return all_data_sent

    def test_badConnect(self):
        """
        A call to the transport's connect method fails with an
        L{InvalidAddressError} when a non-IP address is passed as the host
        value.

        A call to a transport's connect method fails with a L{RuntimeError}
        when the transport is already connected.
        """
        client = GoodClient()
        port = reactor.listenUDP(0, client, interface='127.0.0.1')
        self.assertRaises(error.InvalidAddressError, client.transport.connect, 'localhost', 80)
        client.transport.connect('127.0.0.1', 80)
        self.assertRaises(RuntimeError, client.transport.connect, '127.0.0.1', 80)
        return port.stopListening()

    def test_datagramReceivedError(self):
        """
        When datagramReceived raises an exception it is logged but the port
        is not disconnected.
        """
        finalDeferred = defer.Deferred()

        def cbCompleted(ign):
            """
            Flush the exceptions which the reactor should have logged and make
            sure they're actually there.
            """
            errs = self.flushLoggedErrors(BadClientError)
            self.assertEqual(len(errs), 2, 'Incorrectly found %d errors, expected 2' % (len(errs),))
        finalDeferred.addCallback(cbCompleted)
        client = BadClient()
        port = reactor.listenUDP(0, client, interface='127.0.0.1')

        def cbCleanup(result):
            """
            Disconnect the port we started and pass on whatever was given to us
            in case it was a Failure.
            """
            return defer.maybeDeferred(port.stopListening).addBoth(lambda ign: result)
        finalDeferred.addBoth(cbCleanup)
        addr = port.getHost()
        attempts = list(range(60))
        succeededAttempts = []

        def makeAttempt():
            """
            Send one packet to the listening BadClient.  Set up a 0.1 second
            timeout to do re-transmits in case the packet is dropped.  When two
            packets have been received by the BadClient, stop sending and let
            the finalDeferred's callbacks do some assertions.
            """
            if not attempts:
                try:
                    self.fail('Not enough packets received')
                except Exception:
                    finalDeferred.errback()
            self.failIfIdentical(client.transport, None, 'UDP Protocol lost its transport')
            packet = b'%d' % (attempts.pop(0),)
            packetDeferred = defer.Deferred()
            client.setDeferred(packetDeferred)
            client.transport.write(packet, (addr.host, addr.port))

            def cbPacketReceived(packet):
                """
                A packet arrived.  Cancel the timeout for it, record it, and
                maybe finish the test.
                """
                timeoutCall.cancel()
                succeededAttempts.append(packet)
                if len(succeededAttempts) == 2:
                    reactor.callLater(0, finalDeferred.callback, None)
                else:
                    makeAttempt()

            def ebPacketTimeout(err):
                """
                The packet wasn't received quickly enough.  Try sending another
                one.  It doesn't matter if the packet for which this was the
                timeout eventually arrives: makeAttempt throws away the
                Deferred on which this function is the errback, so when
                datagramReceived callbacks, so it won't be on this Deferred, so
                it won't raise an AlreadyCalledError.
                """
                makeAttempt()
            packetDeferred.addCallbacks(cbPacketReceived, ebPacketTimeout)
            packetDeferred.addErrback(finalDeferred.errback)
            timeoutCall = reactor.callLater(0.1, packetDeferred.errback, error.TimeoutError('Timed out in testDatagramReceivedError'))
        makeAttempt()
        return finalDeferred

    def test_NoWarningOnBroadcast(self):
        """
        C{'<broadcast>'} is an alternative way to say C{'255.255.255.255'}
        ({socket.gethostbyname("<broadcast>")} returns C{'255.255.255.255'}),
        so because it becomes a valid IP address, no deprecation warning about
        passing hostnames to L{twisted.internet.udp.Port.write} needs to be
        emitted by C{write()} in this case.
        """

        class fakeSocket:

            def sendto(self, foo, bar):
                pass
        p = udp.Port(0, Server())
        p.socket = fakeSocket()
        p.write(b'test', ('<broadcast>', 1234))
        warnings = self.flushWarnings([self.test_NoWarningOnBroadcast])
        self.assertEqual(len(warnings), 0)