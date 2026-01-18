import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPFlowStatsReply(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPFlowStatsReply
    """

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = OFPFlowStatsReply(Datapath)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        version = {'buf': b'\x01', 'val': ofproto.OFP_VERSION}
        msg_type = {'buf': b'\x11', 'val': ofproto.OFPT_STATS_REPLY}
        msg_len_val = ofproto.OFP_STATS_MSG_SIZE + ofproto.OFP_FLOW_STATS_SIZE
        msg_len = {'buf': b'\x00d', 'val': msg_len_val}
        xid = {'buf': b'\x94\xc4\xd2\xcd', 'val': 2495926989}
        buf = version['buf'] + msg_type['buf'] + msg_len['buf'] + xid['buf']
        type_ = {'buf': b'\x00\x01', 'val': ofproto.OFPST_FLOW}
        flags = {'buf': b'\x95\xf4', 'val': 38388}
        buf += type_['buf'] + flags['buf']
        length = {'buf': b'\x00`', 'val': 96}
        table_id = {'buf': b'Q', 'val': 81}
        zfill = b'\x00'
        buf += length['buf'] + table_id['buf'] + zfill
        match = b'\x97|\xa6\x1e' + b'^\xa0' + b'p\x17\xdc\x80Y\x9e' + b'y\xc6V\x87\x92(' + b'\xb1\x81' + b'\xbe' + b'\x00' + b'\x01\xab' + b'B' + b'\xfe' + b'\x00\x00' + b'\xa4]\\B' + b'\xa2\\.\x05' + b'Z\x94' + b'd\xd4'
        buf += match
        duration_sec = {'buf': b'\x94\x19\xb3\xd2', 'val': 2484712402}
        duration_nsec = {'buf': b'\xeef\xcf|', 'val': 3999715196}
        priority = {'buf': b'\xe1\xc0', 'val': 57792}
        idle_timeout = {'buf': b'\x8e\x10', 'val': 36368}
        hard_timeout = {'buf': b'\xd4\x99', 'val': 54425}
        zfill = b'\x00' * 6
        cookie = {'buf': b'\x0b\x01\xe8\xe5\xf0\x84\x8a\xe0', 'val': 793171083674290912}
        packet_count = {'buf': b'G\\\xc6\x05(\xff|\xdb', 'val': 5142202600015232219}
        byte_count = {'buf': b'$\xe9K\xee\xcbW\xd9\xc3', 'val': 2659740543924820419}
        buf += duration_sec['buf']
        buf += duration_nsec['buf']
        buf += priority['buf']
        buf += idle_timeout['buf']
        buf += hard_timeout['buf']
        buf += zfill
        buf += cookie['buf']
        buf += packet_count['buf']
        buf += byte_count['buf']
        type = {'buf': b'\x00\x00', 'val': ofproto.OFPAT_OUTPUT}
        len = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_OUTPUT_SIZE}
        port = {'buf': b'Y*', 'val': 22826}
        max_len = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_OUTPUT_SIZE}
        buf += type['buf'] + len['buf'] + port['buf'] + max_len['buf']
        res = OFPFlowStatsReply.parser(object, version['val'], msg_type['val'], msg_len['val'], xid['val'], buf)
        self.assertEqual(version['val'], res.version)
        self.assertEqual(msg_type['val'], res.msg_type)
        self.assertEqual(msg_len['val'], res.msg_len)
        self.assertEqual(xid['val'], res.xid)
        self.assertEqual(type_['val'], res.type)
        self.assertEqual(flags['val'], res.flags)
        body = res.body[0]
        self.assertEqual(length['val'], body.length)
        self.assertEqual(table_id['val'], body.table_id)
        self.assertEqual(duration_sec['val'], body.duration_sec)
        self.assertEqual(duration_nsec['val'], body.duration_nsec)
        self.assertEqual(priority['val'], body.priority)
        self.assertEqual(idle_timeout['val'], body.idle_timeout)
        self.assertEqual(hard_timeout['val'], body.hard_timeout)
        self.assertEqual(cookie['val'], body.cookie)
        self.assertEqual(packet_count['val'], body.packet_count)
        self.assertEqual(byte_count['val'], body.byte_count)
        action = body.actions[0]
        self.assertEqual(type['val'], action.type)
        self.assertEqual(len['val'], action.len)
        self.assertEqual(port['val'], action.port)
        self.assertEqual(max_len['val'], action.max_len)

    def test_serialize(self):
        pass