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
class TestOFPGroupFeaturesStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPGroupFeaturesStats
    """
    types = ofproto.OFPGT_ALL
    capabilities = ofproto.OFPGFC_SELECT_WEIGHT
    max_groups = [1, 2, 3, 4]
    actions = [1 << ofproto.OFPAT_OUTPUT, 1 << ofproto.OFPAT_COPY_TTL_OUT, 1 << ofproto.OFPAT_SET_MPLS_TTL, 1 << ofproto.OFPAT_PUSH_VLAN]

    def test_init(self):
        c = OFPGroupFeaturesStats(self.types, self.capabilities, self.max_groups, self.actions)
        self.assertEqual(self.types, c.types)
        self.assertEqual(self.capabilities, c.capabilities)
        self.assertEqual(self.max_groups, c.max_groups)
        self.assertEqual(self.actions, c.actions)

    def _test_parser(self, types, capabilities, max_groups, actions):
        buf = pack('!I', types) + pack('!I', capabilities) + pack('!I', max_groups[0]) + pack('!I', max_groups[1]) + pack('!I', max_groups[2]) + pack('!I', max_groups[3]) + pack('!I', actions[0]) + pack('!I', actions[1]) + pack('!I', actions[2]) + pack('!I', actions[3])
        res = OFPGroupFeaturesStats.parser(buf, 0)
        self.assertEqual(types, res.types)
        self.assertEqual(capabilities, res.capabilities)
        self.assertEqual(max_groups, res.max_groups)
        self.assertEqual(actions, res.actions)

    def test_parser_mid(self):
        self._test_parser(self.types, self.capabilities, self.max_groups, self.actions)

    def test_parser_max(self):
        types = 4294967295
        capabilities = 4294967295
        max_groups = [4294967295] * 4
        actions = [4294967295] * 4
        self._test_parser(types, capabilities, max_groups, actions)

    def test_parser_min(self):
        types = 0
        capabilities = 0
        max_groups = [0] * 4
        actions = [0] * 4
        self._test_parser(types, capabilities, max_groups, actions)

    def _test_parser_p(self, types, capabilities, actions):
        self._test_parser(types, capabilities, self.max_groups, actions)

    def test_parser_p1(self):
        actions = [1 << ofproto.OFPAT_COPY_TTL_IN, 1 << ofproto.OFPAT_DEC_MPLS_TTL, 1 << ofproto.OFPAT_POP_VLAN, 1 << ofproto.OFPAT_PUSH_MPLS]
        self._test_parser_p(1 << ofproto.OFPGT_ALL, ofproto.OFPGFC_CHAINING, actions)

    def test_parser_p2(self):
        actions = [1 << ofproto.OFPAT_POP_MPLS, 1 << ofproto.OFPAT_SET_QUEUE, 1 << ofproto.OFPAT_GROUP, 1 << ofproto.OFPAT_SET_NW_TTL]
        self._test_parser_p(1 << ofproto.OFPGT_SELECT, ofproto.OFPGFC_SELECT_WEIGHT, actions)

    def test_parser_p3(self):
        actions = [1 << ofproto.OFPAT_DEC_NW_TTL, 1 << ofproto.OFPAT_SET_FIELD, 1 << ofproto.OFPAT_GROUP, 1 << ofproto.OFPAT_SET_NW_TTL]
        self._test_parser_p(1 << ofproto.OFPGT_SELECT, ofproto.OFPGFC_SELECT_LIVENESS, actions)

    def test_parser_p4(self):
        self._test_parser_p(1 << ofproto.OFPGT_INDIRECT, ofproto.OFPGFC_CHAINING, self.actions)

    def test_parser_p5(self):
        self._test_parser_p(1 << ofproto.OFPGT_FF, ofproto.OFPGFC_CHAINING_CHECKS, self.actions)