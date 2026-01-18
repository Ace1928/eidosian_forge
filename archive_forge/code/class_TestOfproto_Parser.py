import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
class TestOfproto_Parser(unittest.TestCase):

    def setUp(self):
        LOG.debug('setUp')
        self.bufHello = binascii.unhexlify('0100000800000001')
        fr = '010600b0000000020000000000000abc' + '00000100010000000000008700000fff' + '0002aefa39d2b9177472656d61302d30' + '00000000000000000000000000000000' + '000000c0000000000000000000000000' + 'fffe723f9a764cc87673775f30786162' + '63000000000000000000000100000001' + '00000082000000000000000000000000' + '00012200d6c5a1947472656d61312d30' + '00000000000000000000000000000000' + '000000c0000000000000000000000000'
        self.bufFeaturesReply = binascii.unhexlify(fr)
        pi = '010a005200000000000001010040' + '00020000000000000002000000000001' + '080045000032000000004011f967c0a8' + '0001c0a8000200010001001e00000000' + '00000000000000000000000000000000' + '00000000'
        self.bufPacketIn = binascii.unhexlify(pi)

    def tearDown(self):
        LOG.debug('tearDown')
        pass

    def testHello(self):
        version, msg_type, msg_len, xid = ofproto_parser.header(self.bufHello)
        self.assertEqual(version, 1)
        self.assertEqual(msg_type, 0)
        self.assertEqual(msg_len, 8)
        self.assertEqual(xid, 1)

    def testFeaturesReply(self):
        version, msg_type, msg_len, xid = ofproto_parser.header(self.bufFeaturesReply)
        msg = ofproto_parser.msg(self, version, msg_type, msg_len, xid, self.bufFeaturesReply)
        LOG.debug(msg)
        self.assertTrue(isinstance(msg, ofproto_v1_0_parser.OFPSwitchFeatures))
        LOG.debug(msg.ports[65534])
        self.assertTrue(isinstance(msg.ports[1], ofproto_v1_0_parser.OFPPhyPort))
        self.assertTrue(isinstance(msg.ports[2], ofproto_v1_0_parser.OFPPhyPort))
        self.assertTrue(isinstance(msg.ports[65534], ofproto_v1_0_parser.OFPPhyPort))

    def testPacketIn(self):
        version, msg_type, msg_len, xid = ofproto_parser.header(self.bufPacketIn)
        msg = ofproto_parser.msg(self, version, msg_type, msg_len, xid, self.bufPacketIn)
        LOG.debug(msg)
        self.assertTrue(isinstance(msg, ofproto_v1_0_parser.OFPPacketIn))

    def test_check_msg_len(self):
        version, msg_type, msg_len, xid = ofproto_parser.header(self.bufPacketIn)
        msg_len = len(self.bufPacketIn) + 1
        self.assertRaises(AssertionError, ofproto_parser.msg, self, version, msg_type, msg_len, xid, self.bufPacketIn)

    def test_check_msg_parser(self):
        version, msg_type, msg_len, xid = ofproto_parser.header(self.bufPacketIn)
        version = 255
        self.assertRaises(exception.OFPUnknownVersion, ofproto_parser.msg, self, version, msg_type, msg_len, xid, self.bufPacketIn)