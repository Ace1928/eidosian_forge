import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_3_parser import *
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_protocol
def _test_set_vlan_vid_none(self):
    header = ofproto.OXM_OF_VLAN_VID
    match = OFPMatch()
    match.set_vlan_vid_none()
    value = ofproto.OFPVID_NONE
    cls_ = OFPMatchField._FIELDS_HEADERS.get(header)
    pack_str = cls_.pack_str.replace('!', '')
    fmt = '!HHI' + pack_str
    buf = bytearray()
    length = match.serialize(buf, 0)
    self.assertEqual(length, len(buf))
    res = list(unpack_from(fmt, bytes(buf), 0)[3:])
    res_value = res.pop(0)
    self.assertEqual(res_value, value)
    res = match.parser(bytes(buf), 0)
    self.assertEqual(res.type, ofproto.OFPMT_OXM)
    self.assertEqual(res.fields[0].header, header)
    self.assertEqual(res.fields[0].value, value)
    jsondict = match.to_jsondict()
    match2 = match.from_jsondict(jsondict['OFPMatch'])
    buf2 = bytearray()
    match2.serialize(buf2, 0)
    self.assertEqual(str(match), str(match2))
    self.assertEqual(buf, buf2)