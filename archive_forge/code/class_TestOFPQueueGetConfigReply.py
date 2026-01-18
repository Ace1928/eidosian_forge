import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
class TestOFPQueueGetConfigReply(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPQueueGetConfigReply
    """

    def _test_parser(self, xid, port, queue_cnt):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_QUEUE_GET_CONFIG_REPLY
        queues_len = 0
        for q in range(queue_cnt):
            queues_len += ofproto.OFP_PACKET_QUEUE_SIZE
            queues_len += ofproto.OFP_QUEUE_PROP_MIN_RATE_SIZE
        msg_len = ofproto.OFP_QUEUE_GET_CONFIG_REPLY_SIZE + queues_len
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        fmt = ofproto.OFP_QUEUE_GET_CONFIG_REPLY_PACK_STR
        buf += pack(fmt, port)
        queues = []
        for q in range(1, queue_cnt + 1):
            fmt = ofproto.OFP_PACKET_QUEUE_PACK_STR
            queue_id = q * 100
            queue_port = q
            queue_len = ofproto.OFP_PACKET_QUEUE_SIZE + ofproto.OFP_QUEUE_PROP_MIN_RATE_SIZE
            buf += pack(fmt, queue_id, queue_port, queue_len)
            fmt = ofproto.OFP_QUEUE_PROP_HEADER_PACK_STR
            prop_type = ofproto.OFPQT_MIN_RATE
            prop_len = ofproto.OFP_QUEUE_PROP_MIN_RATE_SIZE
            buf += pack(fmt, prop_type, prop_len)
            fmt = ofproto.OFP_QUEUE_PROP_MIN_RATE_PACK_STR
            prop_rate = q * 10
            buf += pack(fmt, prop_rate)
            queue = {'queue_id': queue_id, 'queue_port': queue_port, 'queue_len': queue_len, 'prop_type': prop_type, 'prop_len': prop_len, 'prop_rate': prop_rate}
            queues.append(queue)
        res = OFPQueueGetConfigReply.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(port, res.port)
        self.assertEqual(queue_cnt, len(res.queues))
        for i, val in enumerate(res.queues):
            c = queues[i]
            self.assertEqual(c['queue_id'], val.queue_id)
            self.assertEqual(c['queue_port'], val.port)
            self.assertEqual(c['queue_len'], val.len)
            self.assertEqual(1, len(val.properties))
            prop = val.properties[0]
            self.assertEqual(c['prop_type'], prop.property)
            self.assertEqual(c['prop_len'], prop.len)
            self.assertEqual(c['prop_rate'], prop.rate)

    def test_parser_mid(self):
        self._test_parser(2495926989, 65037, 2)

    def test_parser_max(self):
        self._test_parser(4294967295, 4294967295, 2047)

    def test_parser_min(self):
        self._test_parser(0, 0, 0)