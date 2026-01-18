import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPTableStatsReply(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPTableStatsReply
    """

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = OFPTableStatsReply(Datapath)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        version = {'buf': b'\x01', 'val': ofproto.OFP_VERSION}
        msg_type = {'buf': b'\x11', 'val': ofproto.OFPT_STATS_REPLY}
        msg_len_val = ofproto.OFP_STATS_MSG_SIZE + ofproto.OFP_TABLE_STATS_SIZE
        msg_len = {'buf': b'\x00L', 'val': msg_len_val}
        xid = {'buf': b'\xd6\xb4\x8d\xe6', 'val': 3602157030}
        buf = version['buf'] + msg_type['buf'] + msg_len['buf'] + xid['buf']
        type_ = {'buf': b'\x00\x03', 'val': ofproto.OFPST_TABLE}
        flags = {'buf': b'\xb3\xf0', 'val': 46064}
        buf += type_['buf'] + flags['buf']
        table_id = {'buf': b'[', 'val': 91}
        zfill = b'\x00' * 3
        name = b'name'.ljust(32)
        wildcards = {'buf': b'\xc5\xafn\x12', 'val': 3316608530}
        max_entries = {'buf': b'\x95lxM', 'val': 2506913869}
        active_count = {'buf': b'x\xac\xa8\x1e', 'val': 2024581150}
        lookup_count = {'buf': b'@\x1d\x9c9\x19\xec\xd4\x1c', 'val': 4620020561814017052}
        matched_count = {'buf': b"'5\x02\xb6\xc5^\x17e", 'val': 2825167325263435621}
        buf += table_id['buf'] + zfill + name + wildcards['buf'] + max_entries['buf'] + active_count['buf'] + lookup_count['buf'] + matched_count['buf']
        res = OFPTableStatsReply.parser(object, version['val'], msg_type['val'], msg_len['val'], xid['val'], buf)
        self.assertEqual(version['val'], res.version)
        self.assertEqual(msg_type['val'], res.msg_type)
        self.assertEqual(msg_len['val'], res.msg_len)
        self.assertEqual(xid['val'], res.xid)
        self.assertEqual(type_['val'], res.type)
        self.assertEqual(flags['val'], res.flags)
        body = res.body[0]
        self.assertEqual(table_id['val'], body.table_id)
        self.assertEqual(name, body.name)
        self.assertEqual(wildcards['val'], body.wildcards)
        self.assertEqual(max_entries['val'], body.max_entries)
        self.assertEqual(active_count['val'], body.active_count)
        self.assertEqual(lookup_count['val'], body.lookup_count)
        self.assertEqual(matched_count['val'], body.matched_count)

    def test_serialize(self):
        pass