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
class TestOFPErrorMsg(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPErrorMsg
    """
    version = ofproto.OFP_VERSION
    msg_type = ofproto.OFPT_ERROR
    msg_len = ofproto.OFP_ERROR_MSG_SIZE
    xid = 2495926989
    fmt = ofproto.OFP_HEADER_PACK_STR
    buf = pack(fmt, version, msg_type, msg_len, xid)

    def test_init(self):
        c = OFPErrorMsg(_Datapath)
        self.assertEqual(c.code, None)
        self.assertEqual(c.type, None)
        self.assertEqual(c.data, None)

    def _test_parser(self, type_, code, data=None):
        fmt = ofproto.OFP_ERROR_MSG_PACK_STR
        buf = self.buf + pack(fmt, type_, code)
        if data is not None:
            buf += data
        res = OFPErrorMsg.parser(object, self.version, self.msg_type, self.msg_len, self.xid, buf)
        self.assertEqual(res.version, self.version)
        self.assertEqual(res.msg_type, self.msg_type)
        self.assertEqual(res.msg_len, self.msg_len)
        self.assertEqual(res.xid, self.xid)
        self.assertEqual(res.type, type_)
        self.assertEqual(res.code, code)
        if data is not None:
            self.assertEqual(res.data, data)

    def test_parser_mid(self):
        type_ = 32768
        code = 32768
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_max(self):
        type_ = 65534
        code = 65535
        data = b'Error Message.'.ljust(65523)
        self._test_parser(type_, code, data)

    def test_parser_min(self):
        type_ = 0
        code = 0
        data = None
        self._test_parser(type_, code, data)

    def test_parser_p0_1(self):
        type_ = ofproto.OFPET_HELLO_FAILED
        code = ofproto.OFPHFC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_0(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_VERSION
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_1(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_2(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_STAT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_3(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_EXPERIMENTER
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_4(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_EXP_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_5(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_6(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_LEN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_7(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BUFFER_EMPTY
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_8(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BUFFER_UNKNOWN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_9(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_TABLE_ID
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_10(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_IS_SLAVE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_11(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_PORT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p1_12(self):
        type_ = ofproto.OFPET_BAD_REQUEST
        code = ofproto.OFPBRC_BAD_PACKET
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_0(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_1(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_LEN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_2(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_EXPERIMENTER
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_3(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_EXP_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_4(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_OUT_PORT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_5(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_ARGUMENT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_6(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_7(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_TOO_MANY
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_8(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_QUEUE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_9(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_OUT_GROUP
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_10(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_MATCH_INCONSISTENT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_11(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_UNSUPPORTED_ORDER
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_12(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_TAG
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_13(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_SET_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_14(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_SET_LEN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p2_15(self):
        type_ = ofproto.OFPET_BAD_ACTION
        code = ofproto.OFPBAC_BAD_SET_ARGUMENT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_0(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_UNKNOWN_INST
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_1(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_UNSUP_INST
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_2(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_BAD_TABLE_ID
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_3(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_UNSUP_METADATA
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_4(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_UNSUP_METADATA_MASK
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_5(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_BAD_EXPERIMENTER
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_6(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_BAD_EXP_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_7(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_BAD_LEN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p3_8(self):
        type_ = ofproto.OFPET_BAD_INSTRUCTION
        code = ofproto.OFPBIC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_0(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_1(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_LEN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_2(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_TAG
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_3(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_DL_ADDR_MASK
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_4(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_NW_ADDR_MASK
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_5(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_WILDCARDS
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_6(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_FIELD
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_7(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_VALUE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_8(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_MASK
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_9(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_BAD_PREREQ
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_10(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_DUP_FIELD
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p4_11(self):
        type_ = ofproto.OFPET_BAD_MATCH
        code = ofproto.OFPBMC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_0(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_UNKNOWN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_1(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_TABLE_FULL
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_2(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_BAD_TABLE_ID
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_3(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_OVERLAP
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_4(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_5(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_BAD_TIMEOUT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_6(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_BAD_COMMAND
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p5_7(self):
        type_ = ofproto.OFPET_FLOW_MOD_FAILED
        code = ofproto.OFPFMFC_BAD_FLAGS
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_0(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_GROUP_EXISTS
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_1(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_INVALID_GROUP
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_2(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_WEIGHT_UNSUPPORTED
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_3(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_OUT_OF_GROUPS
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_4(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_OUT_OF_BUCKETS
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_5(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_CHAINING_UNSUPPORTED
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_6(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_WATCH_UNSUPPORTED
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_7(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_LOOP
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_8(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_UNKNOWN_GROUP
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_9(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_CHAINED_GROUP
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_10(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_BAD_TYPE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_11(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_BAD_COMMAND
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_12(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_BAD_BUCKET
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_13(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_BAD_WATCH
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p6_14(self):
        type_ = ofproto.OFPET_GROUP_MOD_FAILED
        code = ofproto.OFPGMFC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p7_0(self):
        type_ = ofproto.OFPET_PORT_MOD_FAILED
        code = ofproto.OFPPMFC_BAD_PORT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p7_1(self):
        type_ = ofproto.OFPET_PORT_MOD_FAILED
        code = ofproto.OFPPMFC_BAD_HW_ADDR
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p7_2(self):
        type_ = ofproto.OFPET_PORT_MOD_FAILED
        code = ofproto.OFPPMFC_BAD_CONFIG
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p7_3(self):
        type_ = ofproto.OFPET_PORT_MOD_FAILED
        code = ofproto.OFPPMFC_BAD_ADVERTISE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p7_4(self):
        type_ = ofproto.OFPET_PORT_MOD_FAILED
        code = ofproto.OFPPMFC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p8_0(self):
        type_ = ofproto.OFPET_TABLE_MOD_FAILED
        code = ofproto.OFPTMFC_BAD_TABLE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p8_1(self):
        type_ = ofproto.OFPET_TABLE_MOD_FAILED
        code = ofproto.OFPTMFC_BAD_CONFIG
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p8_2(self):
        type_ = ofproto.OFPET_TABLE_MOD_FAILED
        code = ofproto.OFPTMFC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p9_0(self):
        type_ = ofproto.OFPET_QUEUE_OP_FAILED
        code = ofproto.OFPQOFC_BAD_PORT
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p9_1(self):
        type_ = ofproto.OFPET_QUEUE_OP_FAILED
        code = ofproto.OFPQOFC_BAD_QUEUE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p9_2(self):
        type_ = ofproto.OFPET_QUEUE_OP_FAILED
        code = ofproto.OFPQOFC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p10_0(self):
        type_ = ofproto.OFPET_SWITCH_CONFIG_FAILED
        code = ofproto.OFPSCFC_BAD_FLAGS
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p10_1(self):
        type_ = ofproto.OFPET_SWITCH_CONFIG_FAILED
        code = ofproto.OFPSCFC_BAD_LEN
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p10_2(self):
        type_ = ofproto.OFPET_SWITCH_CONFIG_FAILED
        code = ofproto.OFPSCFC_EPERM
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p11_0(self):
        type_ = ofproto.OFPET_ROLE_REQUEST_FAILED
        code = ofproto.OFPRRFC_STALE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p11_1(self):
        type_ = ofproto.OFPET_ROLE_REQUEST_FAILED
        code = ofproto.OFPRRFC_UNSUP
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_p11_2(self):
        type_ = ofproto.OFPET_ROLE_REQUEST_FAILED
        code = ofproto.OFPRRFC_BAD_ROLE
        data = b'Error Message.'
        self._test_parser(type_, code, data)

    def test_parser_experimenter(self):
        type_ = 65535
        exp_type = 1
        experimenter = 1
        data = b'Error Experimenter Message.'
        fmt = ofproto.OFP_ERROR_EXPERIMENTER_MSG_PACK_STR
        buf = self.buf + pack(fmt, type_, exp_type, experimenter) + data
        res = OFPErrorMsg.parser(object, self.version, self.msg_type, self.msg_len, self.xid, buf)
        self.assertEqual(res.version, self.version)
        self.assertEqual(res.msg_type, self.msg_type)
        self.assertEqual(res.msg_len, self.msg_len)
        self.assertEqual(res.xid, self.xid)
        self.assertEqual(res.type, type_)
        self.assertEqual(res.exp_type, exp_type)
        self.assertEqual(res.experimenter, experimenter)
        self.assertEqual(res.data, data)

    def _test_serialize(self, type_, code, data):
        fmt = ofproto.OFP_ERROR_MSG_PACK_STR
        buf = self.buf + pack(fmt, type_, code) + data
        c = OFPErrorMsg(_Datapath)
        c.type = type_
        c.code = code
        c.data = data
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_ERROR, c.msg_type)
        self.assertEqual(0, c.xid)
        self.assertEqual(len(buf), c.msg_len)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_ERROR_MSG_PACK_STR.replace('!', '') + str(len(c.data)) + 's'
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_ERROR)
        self.assertEqual(res[2], len(buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], type_)
        self.assertEqual(res[5], code)
        self.assertEqual(res[6], data)

    def test_serialize_mid(self):
        type_ = 32768
        code = 32768
        data = b'Error Message.'
        self._test_serialize(type_, code, data)

    def test_serialize_max(self):
        type_ = 65534
        code = 65535
        data = b'Error Message.'.ljust(65523)
        self._test_serialize(type_, code, data)

    def test_serialize_min_except_data(self):
        type_ = ofproto.OFPET_HELLO_FAILED
        code = ofproto.OFPHFC_INCOMPATIBLE
        data = b'Error Message.'
        self._test_serialize(type_, code, data)

    def test_serialize_check_data(self):
        c = OFPErrorMsg(_Datapath)
        self.assertRaises(AssertionError, c.serialize)

    def _test_serialize_p(self, type_, code):
        self._test_serialize(type_, code, b'Error Message.')

    def test_serialize_p0_1(self):
        self._test_serialize_p(ofproto.OFPET_HELLO_FAILED, ofproto.OFPHFC_EPERM)

    def test_serialize_p1_0(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_VERSION)

    def test_serialize_p1_1(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_TYPE)

    def test_serialize_p1_2(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_STAT)

    def test_serialize_p1_3(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_EXPERIMENTER)

    def test_serialize_p1_4(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_EXP_TYPE)

    def test_serialize_p1_5(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_EPERM)

    def test_serialize_p1_6(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_LEN)

    def test_serialize_p1_7(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BUFFER_EMPTY)

    def test_serialize_p1_8(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BUFFER_UNKNOWN)

    def test_serialize_p1_9(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_TABLE_ID)

    def test_serialize_p1_10(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_IS_SLAVE)

    def test_serialize_p1_11(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_PORT)

    def test_serialize_p1_12(self):
        self._test_serialize_p(ofproto.OFPET_BAD_REQUEST, ofproto.OFPBRC_BAD_PACKET)

    def test_serialize_p2_0(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_TYPE)

    def test_serialize_p2_1(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_LEN)

    def test_serialize_p2_2(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_EXPERIMENTER)

    def test_serialize_p2_3(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_EXP_TYPE)

    def test_serialize_p2_4(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_OUT_PORT)

    def test_serialize_p2_5(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_ARGUMENT)

    def test_serialize_p2_6(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_EPERM)

    def test_serialize_p2_7(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_TOO_MANY)

    def test_serialize_p2_8(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_QUEUE)

    def test_serialize_p2_9(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_OUT_GROUP)

    def test_serialize_p2_10(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_MATCH_INCONSISTENT)

    def test_serialize_p2_11(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_UNSUPPORTED_ORDER)

    def test_serialize_p2_12(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_TAG)

    def test_serialize_p2_13(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_SET_TYPE)

    def test_serialize_p2_14(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_SET_LEN)

    def test_serialize_p2_15(self):
        self._test_serialize_p(ofproto.OFPET_BAD_ACTION, ofproto.OFPBAC_BAD_SET_ARGUMENT)

    def test_serialize_p3_0(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_UNKNOWN_INST)

    def test_serialize_p3_1(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_UNSUP_INST)

    def test_serialize_p3_2(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_BAD_TABLE_ID)

    def test_serialize_p3_3(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_UNSUP_METADATA)

    def test_serialize_p3_4(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_UNSUP_METADATA_MASK)

    def test_serialize_p3_5(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_BAD_EXPERIMENTER)

    def test_serialize_p3_6(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_BAD_EXP_TYPE)

    def test_serialize_p3_7(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_BAD_LEN)

    def test_serialize_p3_8(self):
        self._test_serialize_p(ofproto.OFPET_BAD_INSTRUCTION, ofproto.OFPBIC_EPERM)

    def test_serialize_p4_0(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_TYPE)

    def test_serialize_p4_1(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_LEN)

    def test_serialize_p4_2(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_TAG)

    def test_serialize_p4_3(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_DL_ADDR_MASK)

    def test_serialize_p4_4(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_NW_ADDR_MASK)

    def test_serialize_p4_5(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_WILDCARDS)

    def test_serialize_p4_6(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_FIELD)

    def test_serialize_p4_7(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_VALUE)

    def test_serialize_p4_8(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_MASK)

    def test_serialize_p4_9(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_BAD_PREREQ)

    def test_serialize_p4_10(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_DUP_FIELD)

    def test_serialize_p4_11(self):
        self._test_serialize_p(ofproto.OFPET_BAD_MATCH, ofproto.OFPBMC_EPERM)

    def test_serialize_p5_0(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_UNKNOWN)

    def test_serialize_p5_1(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_TABLE_FULL)

    def test_serialize_p5_2(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_BAD_TABLE_ID)

    def test_serialize_p5_3(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_OVERLAP)

    def test_serialize_p5_4(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_EPERM)

    def test_serialize_p5_5(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_BAD_TIMEOUT)

    def test_serialize_p5_6(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_BAD_COMMAND)

    def test_serialize_p5_7(self):
        self._test_serialize_p(ofproto.OFPET_FLOW_MOD_FAILED, ofproto.OFPFMFC_BAD_FLAGS)

    def test_serialize_p6_0(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_GROUP_EXISTS)

    def test_serialize_p6_1(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_INVALID_GROUP)

    def test_serialize_p6_2(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_WEIGHT_UNSUPPORTED)

    def test_serialize_p6_3(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_OUT_OF_GROUPS)

    def test_serialize_p6_4(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_OUT_OF_BUCKETS)

    def test_serialize_p6_5(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_CHAINING_UNSUPPORTED)

    def test_serialize_p6_6(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_WATCH_UNSUPPORTED)

    def test_serialize_p6_7(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_LOOP)

    def test_serialize_p6_8(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_UNKNOWN_GROUP)

    def test_serialize_p6_9(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_CHAINED_GROUP)

    def test_serialize_p6_10(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_BAD_TYPE)

    def test_serialize_p6_11(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_BAD_COMMAND)

    def test_serialize_p6_12(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_BAD_BUCKET)

    def test_serialize_p6_13(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_BAD_WATCH)

    def test_serialize_p6_14(self):
        self._test_serialize_p(ofproto.OFPET_GROUP_MOD_FAILED, ofproto.OFPGMFC_EPERM)

    def test_serialize_p7_0(self):
        self._test_serialize_p(ofproto.OFPET_PORT_MOD_FAILED, ofproto.OFPPMFC_BAD_PORT)

    def test_serialize_p7_1(self):
        self._test_serialize_p(ofproto.OFPET_PORT_MOD_FAILED, ofproto.OFPPMFC_BAD_HW_ADDR)

    def test_serialize_p7_2(self):
        self._test_serialize_p(ofproto.OFPET_PORT_MOD_FAILED, ofproto.OFPPMFC_BAD_CONFIG)

    def test_serialize_p7_3(self):
        self._test_serialize_p(ofproto.OFPET_PORT_MOD_FAILED, ofproto.OFPPMFC_BAD_ADVERTISE)

    def test_serialize_p7_4(self):
        self._test_serialize_p(ofproto.OFPET_PORT_MOD_FAILED, ofproto.OFPPMFC_EPERM)

    def test_serialize_p8_0(self):
        self._test_serialize_p(ofproto.OFPET_TABLE_MOD_FAILED, ofproto.OFPTMFC_BAD_TABLE)

    def test_serialize_p8_1(self):
        self._test_serialize_p(ofproto.OFPET_TABLE_MOD_FAILED, ofproto.OFPTMFC_BAD_CONFIG)

    def test_serialize_p8_2(self):
        self._test_serialize_p(ofproto.OFPET_TABLE_MOD_FAILED, ofproto.OFPTMFC_EPERM)

    def test_serialize_p9_0(self):
        self._test_serialize_p(ofproto.OFPET_QUEUE_OP_FAILED, ofproto.OFPQOFC_BAD_PORT)

    def test_serialize_p9_1(self):
        self._test_serialize_p(ofproto.OFPET_QUEUE_OP_FAILED, ofproto.OFPQOFC_BAD_QUEUE)

    def test_serialize_p9_2(self):
        self._test_serialize_p(ofproto.OFPET_QUEUE_OP_FAILED, ofproto.OFPQOFC_EPERM)

    def test_serialize_p10_0(self):
        self._test_serialize_p(ofproto.OFPET_SWITCH_CONFIG_FAILED, ofproto.OFPSCFC_BAD_FLAGS)

    def test_serialize_p10_1(self):
        self._test_serialize_p(ofproto.OFPET_SWITCH_CONFIG_FAILED, ofproto.OFPSCFC_BAD_LEN)

    def test_serialize_p10_2(self):
        self._test_serialize_p(ofproto.OFPET_SWITCH_CONFIG_FAILED, ofproto.OFPSCFC_EPERM)

    def test_serialize_p11_0(self):
        self._test_serialize_p(ofproto.OFPET_ROLE_REQUEST_FAILED, ofproto.OFPRRFC_STALE)

    def test_serialize_p11_1(self):
        self._test_serialize_p(ofproto.OFPET_ROLE_REQUEST_FAILED, ofproto.OFPRRFC_UNSUP)

    def test_serialize_p11_2(self):
        self._test_serialize_p(ofproto.OFPET_ROLE_REQUEST_FAILED, ofproto.OFPRRFC_BAD_ROLE)