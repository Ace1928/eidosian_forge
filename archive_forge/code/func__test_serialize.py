import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
def _test_serialize(self):

    class Datapath(object):
        ofproto = ofproto_v1_0
        ofproto_parser = ofproto_v1_0_parser
    c = ofproto_v1_0_parser.OFPHello(Datapath)
    c.serialize()
    self.assertEqual(ofproto_v1_0.OFP_VERSION, c.version)
    self.assertEqual(ofproto_v1_0.OFPT_HELLO, c.msg_type)
    self.assertEqual(0, c.xid)
    return True