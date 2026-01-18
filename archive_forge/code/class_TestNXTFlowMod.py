import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXTFlowMod(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXTFlowMod
    """
    cookie = {'buf': b"\x04V'\xad\xbdC\xd6\x83", 'val': 312480851306993283}
    command = {'buf': b'a\xaa', 'val': 25002}
    idle_timeout = {'buf': b'N\xff', 'val': 20223}
    hard_timeout = {'buf': b'\x80\x16', 'val': 32790}
    priority = {'buf': b'p_', 'val': 28767}
    buffer_id = {'buf': b'{\x97:\t', 'val': 2073508361}
    out_port = {'buf': b'\x11}', 'val': 4477}
    flags = {'buf': b'\\\xb9', 'val': 23737}
    rule = nx_match.ClsRule()
    zfill = b'\x00' * 6
    port = {'buf': b'*\xe0', 'val': 10976}
    actions = [OFPActionOutput(port['val'])]

    def _get_obj(self, append_action=False):

        class Datapath(object):
            ofproto = ofproto
            ofproto_parser = ofproto_v1_0_parser
        actions = None
        if append_action:
            actions = self.actions
        c = NXTFlowMod(Datapath, self.cookie['val'], self.command['val'], self.idle_timeout['val'], self.hard_timeout['val'], self.priority['val'], self.buffer_id['val'], self.out_port['val'], self.flags['val'], self.rule, actions)
        return c

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        c = self._get_obj()
        self.assertEqual(self.cookie['val'], c.cookie)
        self.assertEqual(self.command['val'], c.command)
        self.assertEqual(self.idle_timeout['val'], c.idle_timeout)
        self.assertEqual(self.hard_timeout['val'], c.hard_timeout)
        self.assertEqual(self.priority['val'], c.priority)
        self.assertEqual(self.buffer_id['val'], c.buffer_id)
        self.assertEqual(self.out_port['val'], c.out_port)
        self.assertEqual(self.flags['val'], c.flags)
        self.assertEqual(self.rule.__hash__(), c.rule.__hash__())

    def test_init_append_actions(self):
        c = self._get_obj(True)
        action = c.actions[0]
        self.assertEqual(ofproto.OFPAT_OUTPUT, action.type)
        self.assertEqual(ofproto.OFP_ACTION_OUTPUT_SIZE, action.len)
        self.assertEqual(self.port['val'], action.port)

    def test_parser(self):
        pass

    def test_serialize(self):
        c = self._get_obj()
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_VENDOR, c.msg_type)
        self.assertEqual(0, c.xid)
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, c.vendor)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.NICIRA_HEADER_PACK_STR.replace('!', '') + ofproto.NX_FLOW_MOD_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_VENDOR, res[1])
        self.assertEqual(len(c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, res[4])
        self.assertEqual(ofproto.NXT_FLOW_MOD, res[5])
        self.assertEqual(self.cookie['val'], res[6])
        self.assertEqual(self.command['val'], res[7])
        self.assertEqual(self.idle_timeout['val'], res[8])
        self.assertEqual(self.hard_timeout['val'], res[9])
        self.assertEqual(self.priority['val'], res[10])
        self.assertEqual(self.buffer_id['val'], res[11])
        self.assertEqual(self.out_port['val'], res[12])
        self.assertEqual(self.flags['val'], res[13])

    def test_serialize_append_actions(self):
        c = self._get_obj(True)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_VENDOR, c.msg_type)
        self.assertEqual(0, c.xid)
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, c.vendor)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.NICIRA_HEADER_PACK_STR.replace('!', '') + ofproto.NX_FLOW_MOD_PACK_STR.replace('!', '') + ofproto.OFP_ACTION_OUTPUT_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_VENDOR, res[1])
        self.assertEqual(len(c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, res[4])
        self.assertEqual(ofproto.NXT_FLOW_MOD, res[5])
        self.assertEqual(self.cookie['val'], res[6])
        self.assertEqual(self.command['val'], res[7])
        self.assertEqual(self.idle_timeout['val'], res[8])
        self.assertEqual(self.hard_timeout['val'], res[9])
        self.assertEqual(self.priority['val'], res[10])
        self.assertEqual(self.buffer_id['val'], res[11])
        self.assertEqual(self.out_port['val'], res[12])
        self.assertEqual(self.flags['val'], res[13])
        self.assertEqual(0, res[14])
        self.assertEqual(ofproto.OFPAT_OUTPUT, res[15])
        self.assertEqual(ofproto.OFP_ACTION_OUTPUT_SIZE, res[16])
        self.assertEqual(self.port['val'], res[17])
        self.assertEqual(65509, res[18])