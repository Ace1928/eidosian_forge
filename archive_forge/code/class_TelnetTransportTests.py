from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class TelnetTransportTests(unittest.TestCase):
    """
    Tests for L{telnet.TelnetTransport}.
    """

    def setUp(self):
        self.p = telnet.TelnetTransport(TestProtocol)
        self.t = proto_helpers.StringTransport()
        self.p.makeConnection(self.t)

    def testRegularBytes(self):
        h = self.p.protocol
        L = [b'here are some bytes la la la', b'some more arrive here', b'lots of bytes to play with', b'la la la', b'ta de da', b'dum']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L))

    def testNewlineHandling(self):
        h = self.p.protocol
        L = [b'here is the first line\r\n', b'here is the second line\r\x00', b'here is the third line\r\n', b'here is the last line\r\x00']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, L[0][:-2] + b'\n' + L[1][:-2] + b'\r' + L[2][:-2] + b'\n' + L[3][:-2] + b'\r')

    def testIACEscape(self):
        h = self.p.protocol
        L = [b'here are some bytes\xff\xff with an embedded IAC', b'and here is a test of a border escape\xff', b'\xff did you get that IAC?']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L).replace(b'\xff\xff', b'\xff'))

    def _simpleCommandTest(self, cmdName):
        h = self.p.protocol
        cmd = telnet.IAC + getattr(telnet, cmdName)
        L = [b"Here's some bytes, tra la la", b'But ono!' + cmd + b' an interrupt']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.calls, [cmdName])
        self.assertEqual(h.data, b''.join(L).replace(cmd, b''))

    def testInterrupt(self):
        self._simpleCommandTest('IP')

    def testEndOfRecord(self):
        self._simpleCommandTest('EOR')

    def testNoOperation(self):
        self._simpleCommandTest('NOP')

    def testDataMark(self):
        self._simpleCommandTest('DM')

    def testBreak(self):
        self._simpleCommandTest('BRK')

    def testAbortOutput(self):
        self._simpleCommandTest('AO')

    def testAreYouThere(self):
        self._simpleCommandTest('AYT')

    def testEraseCharacter(self):
        self._simpleCommandTest('EC')

    def testEraseLine(self):
        self._simpleCommandTest('EL')

    def testGoAhead(self):
        self._simpleCommandTest('GA')

    def testSubnegotiation(self):
        h = self.p.protocol
        cmd = telnet.IAC + telnet.SB + b'\x12hello world' + telnet.IAC + telnet.SE
        L = [b'These are some bytes but soon' + cmd, b'there will be some more']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
        self.assertEqual(h.subcmd, list(iterbytes(b'hello world')))

    def testSubnegotiationWithEmbeddedSE(self):
        h = self.p.protocol
        cmd = telnet.IAC + telnet.SB + b'\x12' + telnet.SE + telnet.IAC + telnet.SE
        L = [b'Some bytes are here' + cmd + b'and here', b'and here']
        for b in L:
            self.p.dataReceived(b)
        self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
        self.assertEqual(h.subcmd, [telnet.SE])

    def testBoundarySubnegotiation(self):
        cmd = telnet.IAC + telnet.SB + b'\x12' + telnet.SE + b'hello' + telnet.IAC + telnet.SE
        for i in range(len(cmd)):
            h = self.p.protocol = TestProtocol()
            h.makeConnection(self.p)
            a, b = (cmd[:i], cmd[i:])
            L = [b'first part' + a, b + b'last part']
            for data in L:
                self.p.dataReceived(data)
            self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
            self.assertEqual(h.subcmd, [telnet.SE] + list(iterbytes(b'hello')))

    def _enabledHelper(self, o, eL=[], eR=[], dL=[], dR=[]):
        self.assertEqual(o.enabledLocal, eL)
        self.assertEqual(o.enabledRemote, eR)
        self.assertEqual(o.disabledLocal, dL)
        self.assertEqual(o.disabledRemote, dR)

    def testRefuseWill(self):
        cmd = telnet.IAC + telnet.WILL + b'\x12'
        data = b'surrounding bytes' + cmd + b'to spice things up'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b'\x12')
        self._enabledHelper(self.p.protocol)

    def testRefuseDo(self):
        cmd = telnet.IAC + telnet.DO + b'\x12'
        data = b'surrounding bytes' + cmd + b'to spice things up'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WONT + b'\x12')
        self._enabledHelper(self.p.protocol)

    def testAcceptDo(self):
        cmd = telnet.IAC + telnet.DO + b'\x19'
        data = b'padding' + cmd + b'trailer'
        h = self.p.protocol
        h.localEnableable = (b'\x19',)
        self.p.dataReceived(data)
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WILL + b'\x19')
        self._enabledHelper(h, eL=[b'\x19'])

    def testAcceptWill(self):
        cmd = telnet.IAC + telnet.WILL + b'\x91'
        data = b'header' + cmd + b'padding'
        h = self.p.protocol
        h.remoteEnableable = (b'\x91',)
        self.p.dataReceived(data)
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'\x91')
        self._enabledHelper(h, eR=[b'\x91'])

    def testAcceptWont(self):
        cmd = telnet.IAC + telnet.WONT + b')'
        s = self.p.getOptionState(b')')
        s.him.state = 'yes'
        data = b'fiddle dee' + cmd
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b')')
        self.assertEqual(s.him.state, 'no')
        self._enabledHelper(self.p.protocol, dR=[b')'])

    def testAcceptDont(self):
        cmd = telnet.IAC + telnet.DONT + b')'
        s = self.p.getOptionState(b')')
        s.us.state = 'yes'
        data = b'fiddle dum ' + cmd
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WONT + b')')
        self.assertEqual(s.us.state, 'no')
        self._enabledHelper(self.p.protocol, dL=[b')'])

    def testIgnoreWont(self):
        cmd = telnet.IAC + telnet.WONT + b'G'
        data = b'dum de dum' + cmd + b'tra la la'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testIgnoreDont(self):
        cmd = telnet.IAC + telnet.DONT + b'G'
        data = b'dum de dum' + cmd + b'tra la la'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testIgnoreWill(self):
        cmd = telnet.IAC + telnet.WILL + b'V'
        s = self.p.getOptionState(b'V')
        s.him.state = 'yes'
        data = b'tra la la' + cmd + b'dum de dum'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testIgnoreDo(self):
        cmd = telnet.IAC + telnet.DO + b'V'
        s = self.p.getOptionState(b'V')
        s.us.state = 'yes'
        data = b'tra la la' + cmd + b'dum de dum'
        self.p.dataReceived(data)
        self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
        self.assertEqual(self.t.value(), b'')
        self._enabledHelper(self.p.protocol)

    def testAcceptedEnableRequest(self):
        d = self.p.do(b'B')
        h = self.p.protocol
        h.remoteEnableable = (b'B',)
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'B')
        self.p.dataReceived(telnet.IAC + telnet.WILL + b'B')
        d.addCallback(self.assertEqual, True)
        d.addCallback(lambda _: self._enabledHelper(h, eR=[b'B']))
        return d

    def test_refusedEnableRequest(self):
        """
        If the peer refuses to enable an option we request it to enable, the
        L{Deferred} returned by L{TelnetProtocol.do} fires with an
        L{OptionRefused} L{Failure}.
        """
        self.p.protocol.remoteEnableable = (b'B',)
        d = self.p.do(b'B')
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'B')
        s = self.p.getOptionState(b'B')
        self.assertEqual(s.him.state, 'no')
        self.assertEqual(s.us.state, 'no')
        self.assertTrue(s.him.negotiating)
        self.assertFalse(s.us.negotiating)
        self.p.dataReceived(telnet.IAC + telnet.WONT + b'B')
        d = self.assertFailure(d, telnet.OptionRefused)
        d.addCallback(lambda ignored: self._enabledHelper(self.p.protocol))
        d.addCallback(lambda ignored: self.assertFalse(s.him.negotiating))
        return d

    def test_refusedEnableOffer(self):
        """
        If the peer refuses to allow us to enable an option, the L{Deferred}
        returned by L{TelnetProtocol.will} fires with an L{OptionRefused}
        L{Failure}.
        """
        self.p.protocol.localEnableable = (b'B',)
        d = self.p.will(b'B')
        self.assertEqual(self.t.value(), telnet.IAC + telnet.WILL + b'B')
        s = self.p.getOptionState(b'B')
        self.assertEqual(s.him.state, 'no')
        self.assertEqual(s.us.state, 'no')
        self.assertFalse(s.him.negotiating)
        self.assertTrue(s.us.negotiating)
        self.p.dataReceived(telnet.IAC + telnet.DONT + b'B')
        d = self.assertFailure(d, telnet.OptionRefused)
        d.addCallback(lambda ignored: self._enabledHelper(self.p.protocol))
        d.addCallback(lambda ignored: self.assertFalse(s.us.negotiating))
        return d

    def testAcceptedDisableRequest(self):
        s = self.p.getOptionState(b'B')
        s.him.state = 'yes'
        d = self.p.dont(b'B')
        self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b'B')
        self.p.dataReceived(telnet.IAC + telnet.WONT + b'B')
        d.addCallback(self.assertEqual, True)
        d.addCallback(lambda _: self._enabledHelper(self.p.protocol, dR=[b'B']))
        return d

    def testNegotiationBlocksFurtherNegotiation(self):
        s = self.p.getOptionState(b'$')
        s.him.state = 'yes'
        self.p.dont(b'$')

        def _do(x):
            d = self.p.do(b'$')
            return self.assertFailure(d, telnet.AlreadyNegotiating)

        def _dont(x):
            d = self.p.dont(b'$')
            return self.assertFailure(d, telnet.AlreadyNegotiating)

        def _final(x):
            self.p.dataReceived(telnet.IAC + telnet.WONT + b'$')
            self._enabledHelper(self.p.protocol, dR=[b'$'])
            self.p.protocol.remoteEnableable = (b'$',)
            d = self.p.do(b'$')
            self.p.dataReceived(telnet.IAC + telnet.WILL + b'$')
            d.addCallback(self.assertEqual, True)
            d.addCallback(lambda _: self._enabledHelper(self.p.protocol, eR=[b'$'], dR=[b'$']))
            return d
        d = _do(None)
        d.addCallback(_dont)
        d.addCallback(_final)
        return d

    def testSuperfluousDisableRequestRaises(self):
        d = self.p.dont(b'\xab')
        return self.assertFailure(d, telnet.AlreadyDisabled)

    def testSuperfluousEnableRequestRaises(self):
        s = self.p.getOptionState(b'\xab')
        s.him.state = 'yes'
        d = self.p.do(b'\xab')
        return self.assertFailure(d, telnet.AlreadyEnabled)

    def testLostConnectionFailsDeferreds(self):
        d1 = self.p.do(b'\x12')
        d2 = self.p.do(b'#')
        d3 = self.p.do(b'4')

        class TestException(Exception):
            pass
        self.p.connectionLost(TestException('Total failure!'))
        d1 = self.assertFailure(d1, TestException)
        d2 = self.assertFailure(d2, TestException)
        d3 = self.assertFailure(d3, TestException)
        return defer.gatherResults([d1, d2, d3])