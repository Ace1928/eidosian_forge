import unittest
import logging
import struct
from os_ken.lib.packet import bpdu
class Test_TopologyChangeNotificationBPDUs(unittest.TestCase):
    """ Test case for TopologyChangeNotificationBPDUs
    """

    def setUp(self):
        self.protocol_id = bpdu.PROTOCOL_IDENTIFIER
        self.version_id = bpdu.TopologyChangeNotificationBPDUs.VERSION_ID
        self.bpdu_type = bpdu.TopologyChangeNotificationBPDUs.BPDU_TYPE
        self.msg = bpdu.TopologyChangeNotificationBPDUs()
        self.fmt = bpdu.bpdu._PACK_STR
        self.buf = struct.pack(self.fmt, self.protocol_id, self.version_id, self.bpdu_type)

    def test_init(self):
        self.assertEqual(self.protocol_id, self.msg._protocol_id)
        self.assertEqual(self.version_id, self.msg._version_id)
        self.assertEqual(self.bpdu_type, self.msg._bpdu_type)

    def test_parser(self):
        r1, r2, _ = bpdu.bpdu.parser(self.buf)
        self.assertEqual(type(r1), type(self.msg))
        self.assertEqual(r1._protocol_id, self.protocol_id)
        self.assertEqual(r1._version_id, self.version_id)
        self.assertEqual(r1._bpdu_type, self.bpdu_type)
        self.assertEqual(r2, None)

    def test_serialize(self):
        data = bytearray()
        prev = None
        buf = self.msg.serialize(data, prev)
        res = struct.unpack(self.fmt, buf)
        self.assertEqual(res[0], self.protocol_id)
        self.assertEqual(res[1], self.version_id)
        self.assertEqual(res[2], self.bpdu_type)

    def test_json(self):
        jsondict = self.msg.to_jsondict()
        msg = bpdu.TopologyChangeNotificationBPDUs.from_jsondict(jsondict['TopologyChangeNotificationBPDUs'])
        self.assertEqual(str(self.msg), str(msg))