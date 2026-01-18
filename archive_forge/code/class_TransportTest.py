from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
class TransportTest(unittest.TestCase):

    def testInit(self):
        fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]))
        t = hidtransport.UsbHidTransport(fake_hid_dev)
        self.assertEquals(t.cid, bytearray([0, 0, 0, 1]))
        self.assertEquals(t.u2fhid_version, 1)

    def testPing(self):
        fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]))
        t = hidtransport.UsbHidTransport(fake_hid_dev)
        reply = t.SendPing(b'1234')
        self.assertEquals(reply, b'1234')

    def testMsg(self):
        fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]), bytearray([1, 144, 0]))
        t = hidtransport.UsbHidTransport(fake_hid_dev)
        reply = t.SendMsgBytes([0, 1, 0, 0])
        self.assertEquals(reply, bytearray([1, 144, 0]))

    def testMsgBusy(self):
        fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]), bytearray([1, 144, 0]))
        t = hidtransport.UsbHidTransport(fake_hid_dev)
        fake_hid_dev.SetChannelBusyCount(3)
        with mock.patch.object(hidtransport, 'time') as _:
            self.assertRaisesRegexp(errors.HidError, '^Device Busy', t.SendMsgBytes, [0, 1, 0, 0])
            reply = t.SendMsgBytes([0, 1, 0, 0])
            self.assertEquals(reply, bytearray([1, 144, 0]))

    def testFragmentedResponseMsg(self):
        body = bytearray([x % 256 for x in range(0, 1000)])
        fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]), body)
        t = hidtransport.UsbHidTransport(fake_hid_dev)
        reply = t.SendMsgBytes([0, 1, 0, 0])
        self.assertEquals(reply, bytearray((x % 256 for x in range(0, 1000))))

    def testFragmentedSendApdu(self):
        body = bytearray((x % 256 for x in range(0, 1000)))
        fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]), [144, 0])
        t = hidtransport.UsbHidTransport(fake_hid_dev)
        reply = t.SendMsgBytes(body)
        self.assertEquals(reply, bytearray([144, 0]))
        self.assertEquals(len(fake_hid_dev.received_packets), 18)

    def testInitPacketShape(self):
        packet = hidtransport.UsbHidTransport.InitPacket(64, bytearray(b'\x00\x00\x00\x01'), 131, 2, bytearray(b'\x01\x02'))
        self.assertEquals(packet.ToWireFormat(), RPad([0, 0, 0, 1, 131, 0, 2, 1, 2], 64))
        copy = hidtransport.UsbHidTransport.InitPacket.FromWireFormat(64, packet.ToWireFormat())
        self.assertEquals(copy.cid, bytearray(b'\x00\x00\x00\x01'))
        self.assertEquals(copy.cmd, 131)
        self.assertEquals(copy.size, 2)
        self.assertEquals(copy.payload, bytearray(b'\x01\x02'))

    def testContPacketShape(self):
        packet = hidtransport.UsbHidTransport.ContPacket(64, bytearray(b'\x00\x00\x00\x01'), 5, bytearray(b'\x01\x02'))
        self.assertEquals(packet.ToWireFormat(), RPad([0, 0, 0, 1, 5, 1, 2], 64))
        copy = hidtransport.UsbHidTransport.ContPacket.FromWireFormat(64, packet.ToWireFormat())
        self.assertEquals(copy.cid, bytearray(b'\x00\x00\x00\x01'))
        self.assertEquals(copy.seq, 5)
        self.assertEquals(copy.payload, RPad(bytearray(b'\x01\x02'), 59))