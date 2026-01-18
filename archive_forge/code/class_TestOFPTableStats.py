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
class TestOFPTableStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPTableStats
    """

    def test_init(self):
        table_id = 91
        name = 'name'
        match = 1270985291017894273
        wildcards = 3316608530
        write_actions = 2484712402
        apply_actions = 3999715196
        write_setfields = 5142202600015232219
        apply_setfields = 2659740543924820419
        metadata_match = 2127614848199081640
        metadata_write = 2127614848199081641
        instructions = 1119692796
        config = 2226555987
        max_entries = 2506913869
        active_count = 2024581150
        lookup_count = 4620020561814017052
        matched_count = 2825167325263435621
        res = OFPTableStats(table_id, name, match, wildcards, write_actions, apply_actions, write_setfields, apply_setfields, metadata_match, metadata_write, instructions, config, max_entries, active_count, lookup_count, matched_count)
        self.assertEqual(table_id, res.table_id)
        self.assertEqual(name, res.name)
        self.assertEqual(match, res.match)
        self.assertEqual(wildcards, res.wildcards)
        self.assertEqual(write_actions, res.write_actions)
        self.assertEqual(apply_actions, res.apply_actions)
        self.assertEqual(write_setfields, res.write_setfields)
        self.assertEqual(apply_setfields, res.apply_setfields)
        self.assertEqual(metadata_match, res.metadata_match)
        self.assertEqual(metadata_write, res.metadata_write)
        self.assertEqual(instructions, res.instructions)
        self.assertEqual(config, res.config)
        self.assertEqual(max_entries, res.max_entries)
        self.assertEqual(active_count, res.active_count)
        self.assertEqual(lookup_count, res.lookup_count)
        self.assertEqual(matched_count, res.matched_count)

    def _test_parser(self, table_id, name, match, wildcards, write_actions, apply_actions, write_setfields, apply_setfields, metadata_match, metadata_write, instructions, config, max_entries, active_count, lookup_count, matched_count):
        fmt = ofproto.OFP_TABLE_STATS_PACK_STR
        buf = pack(fmt, table_id, name, match, wildcards, write_actions, apply_actions, write_setfields, apply_setfields, metadata_match, metadata_write, instructions, config, max_entries, active_count, lookup_count, matched_count)
        res = OFPTableStats.parser(buf, 0)
        self.assertEqual(table_id, res.table_id)
        self.assertEqual(name, res.name.replace(b'\x00', b''))
        self.assertEqual(match, res.match)
        self.assertEqual(wildcards, res.wildcards)
        self.assertEqual(write_actions, res.write_actions)
        self.assertEqual(apply_actions, res.apply_actions)
        self.assertEqual(write_setfields, res.write_setfields)
        self.assertEqual(apply_setfields, res.apply_setfields)
        self.assertEqual(metadata_match, res.metadata_match)
        self.assertEqual(metadata_write, res.metadata_write)
        self.assertEqual(instructions, res.instructions)
        self.assertEqual(config, res.config)
        self.assertEqual(max_entries, res.max_entries)
        self.assertEqual(active_count, res.active_count)
        self.assertEqual(lookup_count, res.lookup_count)
        self.assertEqual(matched_count, res.matched_count)

    def test_parser_mid(self):
        table_id = 91
        name = b'name'
        match = 1270985291017894273
        wildcards = 3316608530
        write_actions = 2484712402
        apply_actions = 3999715196
        write_setfields = 5142202600015232219
        apply_setfields = 2659740543924820419
        metadata_match = 2127614848199081640
        metadata_write = 2127614848199081641
        instructions = 1119692796
        config = 2226555987
        max_entries = 2506913869
        active_count = 2024581150
        lookup_count = 4620020561814017052
        matched_count = 2825167325263435621
        self._test_parser(table_id, name, match, wildcards, write_actions, apply_actions, write_setfields, apply_setfields, metadata_match, metadata_write, instructions, config, max_entries, active_count, lookup_count, matched_count)

    def test_parser_max(self):
        table_id = 255
        name = b'a' * 32
        match = 18446744073709551615
        wildcards = 18446744073709551615
        write_actions = 4294967295
        apply_actions = 4294967295
        write_setfields = 18446744073709551615
        apply_setfields = 18446744073709551615
        metadata_match = 18446744073709551615
        metadata_write = 18446744073709551615
        instructions = 4294967295
        config = 4294967295
        max_entries = 4294967295
        active_count = 4294967295
        lookup_count = 18446744073709551615
        matched_count = 18446744073709551615
        self._test_parser(table_id, name, match, wildcards, write_actions, apply_actions, write_setfields, apply_setfields, metadata_match, metadata_write, instructions, config, max_entries, active_count, lookup_count, matched_count)

    def test_parser_min(self):
        table_id = 0
        name = b''
        match = 0
        wildcards = 0
        write_actions = 0
        apply_actions = 0
        write_setfields = 0
        apply_setfields = 0
        metadata_match = 0
        metadata_write = 0
        instructions = 0
        config = 0
        max_entries = 0
        active_count = 0
        lookup_count = 0
        matched_count = 0
        self._test_parser(table_id, name, match, wildcards, write_actions, apply_actions, write_setfields, apply_setfields, metadata_match, metadata_write, instructions, config, max_entries, active_count, lookup_count, matched_count)

    def _test_parser_p(self, ofpxmt, ofpit, ofptc):
        table_id = 91
        name = b'name'
        match = ofpxmt
        wildcards = ofpxmt
        write_actions = 2484712402
        apply_actions = 3999715196
        write_setfields = ofpxmt
        apply_setfields = ofpxmt
        metadata_match = 2127614848199081640
        metadata_write = 2127614848199081641
        instructions = ofpit
        config = ofptc
        max_entries = 2506913869
        active_count = 2024581150
        lookup_count = 4620020561814017052
        matched_count = 2825167325263435621
        self._test_parser(table_id, name, match, wildcards, write_actions, apply_actions, write_setfields, apply_setfields, metadata_match, metadata_write, instructions, config, max_entries, active_count, lookup_count, matched_count)

    def test_parser_p1(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IN_PORT, ofproto.OFPIT_GOTO_TABLE, ofproto.OFPTC_TABLE_MISS_CONTINUE)

    def test_parser_p2(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IN_PHY_PORT, ofproto.OFPIT_WRITE_METADATA, ofproto.OFPTC_TABLE_MISS_DROP)

    def test_parser_p3(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_METADATA, ofproto.OFPIT_WRITE_ACTIONS, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p4(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ETH_DST, ofproto.OFPIT_APPLY_ACTIONS, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p5(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ETH_SRC, ofproto.OFPIT_CLEAR_ACTIONS, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p6(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ETH_TYPE, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p7(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_VLAN_VID, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p8(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_VLAN_PCP, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p9(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IP_DSCP, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p10(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IP_ECN, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p11(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IP_PROTO, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p12(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV4_SRC, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p13(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV4_DST, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p14(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_TCP_SRC, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p15(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_TCP_DST, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p16(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_UDP_SRC, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p17(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_UDP_DST, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p18(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_SCTP_SRC, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p19(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_SCTP_DST, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p20(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ICMPV4_TYPE, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p21(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ICMPV4_CODE, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p22(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ARP_OP, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p23(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ARP_SPA, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p24(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ARP_TPA, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p25(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ARP_SHA, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p26(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ARP_THA, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p27(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV6_SRC, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p28(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV6_DST, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p29(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV6_FLABEL, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p30(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ICMPV6_TYPE, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p31(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_ICMPV6_CODE, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p32(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV6_ND_TARGET, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p33(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV6_ND_SLL, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p34(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_IPV6_ND_TLL, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p35(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_MPLS_LABEL, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)

    def test_parser_p36(self):
        self._test_parser_p(ofproto.OFPXMT_OFB_MPLS_TC, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)