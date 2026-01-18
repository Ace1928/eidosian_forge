import copy
import logging
from struct import pack, unpack_from
import unittest
from os_ken.ofproto import ether
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib import addrconv
from os_ken.lib.packet.slow import slow, lacp
from os_ken.lib.packet.slow import SLOW_PROTOCOL_MULTICAST
from os_ken.lib.packet.slow import SLOW_SUBTYPE_LACP
from os_ken.lib.packet.slow import SLOW_SUBTYPE_MARKER
class Test_slow(unittest.TestCase):
    """ Test case for Slow Protocol
    """

    def setUp(self):
        self.subtype = SLOW_SUBTYPE_LACP
        self.version = lacp.LACP_VERSION_NUMBER
        self.actor_tag = lacp.LACP_TLV_TYPE_ACTOR
        self.actor_length = 20
        self.actor_system_priority = 65534
        self.actor_system = '00:07:0d:af:f4:54'
        self.actor_key = 1
        self.actor_port_priority = 65535
        self.actor_port = 1
        self.actor_state_activity = lacp.LACP_STATE_ACTIVE
        self.actor_state_timeout = lacp.LACP_STATE_LONG_TIMEOUT
        self.actor_state_aggregation = lacp.LACP_STATE_AGGREGATEABLE
        self.actor_state_synchronization = lacp.LACP_STATE_IN_SYNC
        self.actor_state_collecting = lacp.LACP_STATE_COLLECTING_ENABLED
        self.actor_state_distributing = lacp.LACP_STATE_DISTRIBUTING_ENABLED
        self.actor_state_defaulted = lacp.LACP_STATE_OPERATIONAL_PARTNER
        self.actor_state_expired = lacp.LACP_STATE_EXPIRED
        self.actor_state = self.actor_state_activity << 0 | self.actor_state_timeout << 1 | self.actor_state_aggregation << 2 | self.actor_state_synchronization << 3 | self.actor_state_collecting << 4 | self.actor_state_distributing << 5 | self.actor_state_defaulted << 6 | self.actor_state_expired << 7
        self.partner_tag = lacp.LACP_TLV_TYPE_PARTNER
        self.partner_length = 20
        self.partner_system_priority = 0
        self.partner_system = '00:00:00:00:00:00'
        self.partner_key = 0
        self.partner_port_priority = 0
        self.partner_port = 0
        self.partner_state_activity = 0
        self.partner_state_timeout = lacp.LACP_STATE_SHORT_TIMEOUT
        self.partner_state_aggregation = 0
        self.partner_state_synchronization = 0
        self.partner_state_collecting = 0
        self.partner_state_distributing = 0
        self.partner_state_defaulted = 0
        self.partner_state_expired = 0
        self.partner_state = self.partner_state_activity << 0 | self.partner_state_timeout << 1 | self.partner_state_aggregation << 2 | self.partner_state_synchronization << 3 | self.partner_state_collecting << 4 | self.partner_state_distributing << 5 | self.partner_state_defaulted << 6 | self.partner_state_expired << 7
        self.collector_tag = lacp.LACP_TLV_TYPE_COLLECTOR
        self.collector_length = 16
        self.collector_max_delay = 0
        self.terminator_tag = lacp.LACP_TLV_TYPE_TERMINATOR
        self.terminator_length = 0
        self.head_fmt = lacp._HLEN_PACK_STR
        self.head_len = lacp._HLEN_PACK_LEN
        self.act_fmt = lacp._ACTPRT_INFO_PACK_STR
        self.act_len = lacp._ACTPRT_INFO_PACK_LEN
        self.prt_fmt = lacp._ACTPRT_INFO_PACK_STR
        self.prt_len = lacp._ACTPRT_INFO_PACK_LEN
        self.col_fmt = lacp._COL_INFO_PACK_STR
        self.col_len = lacp._COL_INFO_PACK_LEN
        self.trm_fmt = lacp._TRM_PACK_STR
        self.trm_len = lacp._TRM_PACK_LEN
        self.length = lacp._ALL_PACK_LEN
        self.head_buf = pack(self.head_fmt, self.subtype, self.version)
        self.act_buf = pack(self.act_fmt, self.actor_tag, self.actor_length, self.actor_system_priority, addrconv.mac.text_to_bin(self.actor_system), self.actor_key, self.actor_port_priority, self.actor_port, self.actor_state)
        self.prt_buf = pack(self.prt_fmt, self.partner_tag, self.partner_length, self.partner_system_priority, addrconv.mac.text_to_bin(self.partner_system), self.partner_key, self.partner_port_priority, self.partner_port, self.partner_state)
        self.col_buf = pack(self.col_fmt, self.collector_tag, self.collector_length, self.collector_max_delay)
        self.trm_buf = pack(self.trm_fmt, self.terminator_tag, self.terminator_length)
        self.buf = self.head_buf + self.act_buf + self.prt_buf + self.col_buf + self.trm_buf

    def tearDown(self):
        pass

    def test_parser(self):
        slow.parser(self.buf)

    def test_not_implemented_subtype(self):
        not_implemented_buf = pack(slow._PACK_STR, SLOW_SUBTYPE_MARKER) + self.buf[1:]
        instance, nexttype, last = slow.parser(not_implemented_buf)
        assert instance is None
        assert nexttype is None
        assert last is not None

    def test_invalid_subtype(self):
        invalid_buf = b'\xff' + self.buf[1:]
        instance, nexttype, last = slow.parser(invalid_buf)
        assert instance is None
        assert nexttype is None
        assert last is not None