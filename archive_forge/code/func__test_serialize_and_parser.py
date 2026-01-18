import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_3_parser import *
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_protocol
def _test_serialize_and_parser(self, match, header, value, mask=None):
    cls_ = OFPMatchField._FIELDS_HEADERS.get(header)
    pack_str = cls_.pack_str.replace('!', '')
    fmt = '!HHI' + pack_str
    buf = bytearray()
    length = match.serialize(buf, 0)
    self.assertEqual(length, len(buf))
    if mask and len(buf) > calcsize(fmt):
        fmt += pack_str
    res = list(unpack_from(fmt, bytes(buf), 0)[3:])
    if type(value) is list:
        res_value = res[:calcsize(pack_str) // 2]
        self.assertEqual(res_value, value)
        if mask:
            res_mask = res[calcsize(pack_str) // 2:]
            self.assertEqual(res_mask, mask)
    else:
        res_value = res.pop(0)
        if cls_.__name__ == 'MTVlanVid':
            self.assertEqual(res_value, value | ofproto.OFPVID_PRESENT)
        else:
            self.assertEqual(res_value, value)
        if mask and res and res[0]:
            res_mask = res[0]
            self.assertEqual(res_mask, mask)
    res = match.parser(bytes(buf), 0)
    self.assertEqual(res.type, ofproto.OFPMT_OXM)
    self.assertEqual(res.fields[0].header, header)
    self.assertEqual(res.fields[0].value, value)
    if mask and res.fields[0].mask is not None:
        self.assertEqual(res.fields[0].mask, mask)
    jsondict = match.to_jsondict()
    match2 = match.from_jsondict(jsondict['OFPMatch'])
    buf2 = bytearray()
    match2.serialize(buf2, 0)
    self.assertEqual(str(match), str(match2))
    self.assertEqual(buf, buf2)