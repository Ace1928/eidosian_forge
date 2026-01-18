import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPActionEnqueue(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPActionEnqueue
    """
    type_ = {'buf': b'\x00\x0b', 'val': ofproto.OFPAT_ENQUEUE}
    len_ = {'buf': b'\x00\x10', 'val': ofproto.OFP_ACTION_ENQUEUE_SIZE}
    port = {'buf': b'\x04U', 'val': 1109}
    zfill = b'\x00' * 6
    queue_id = {'buf': b'\n[\x03^', 'val': 173736798}
    buf = type_['buf'] + len_['buf'] + port['buf'] + zfill + queue_id['buf']
    c = OFPActionEnqueue(port['val'], queue_id['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.port['val'], self.c.port)
        self.assertEqual(self.queue_id['val'], self.c.queue_id)

    def test_parser(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.port['val'], res.port)
        self.assertEqual(self.queue_id['val'], res.queue_id)

    def test_parser_check_type(self):
        type_ = {'buf': b'\x00\n', 'val': 10}
        buf = type_['buf'] + self.len_['buf'] + self.port['buf'] + self.zfill + self.queue_id['buf']
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_parser_check_len(self):
        len_ = {'buf': b'\x00\x05', 'val': 5}
        buf = self.type_['buf'] + len_['buf'] + self.port['buf'] + self.zfill + self.queue_id['buf']
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_ENQUEUE_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.port['val'], res[2])
        self.assertEqual(self.queue_id['val'], res[3])