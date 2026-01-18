import unittest
import os_ken.ofproto.ofproto_v1_3_parser as ofpp
class Test_FlowSpec(unittest.TestCase):

    def test_flowspec_src_0_dst_0(self):
        user = ofpp.NXFlowSpecMatch(src=('in_port', 0), dst=('in_port', 0), n_bits=16)
        on_wire = b'\x00\x10\x80\x00\x00\x04\x00\x00\x80\x00\x00\x04\x00\x00'
        self.assertEqual(on_wire, user.serialize())
        o, rest = ofpp._NXFlowSpec.parse(on_wire)
        self.assertEqual(user.to_jsondict(), o.to_jsondict())
        self.assertEqual(str(user), str(o))
        self.assertEqual(b'', rest)

    def test_flowspec_src_1_dst_0(self):
        user = ofpp.NXFlowSpecMatch(src=99, dst=('in_port', 0), n_bits=16)
        on_wire = b' \x10\x00c\x80\x00\x00\x04\x00\x00'
        self.assertEqual(on_wire, user.serialize())
        o, rest = ofpp._NXFlowSpec.parse(on_wire)
        self.assertEqual(user.to_jsondict(), o.to_jsondict())
        self.assertEqual(str(user), str(o))
        self.assertEqual(b'', rest)

    def test_flowspec_src_0_dst_1(self):
        user = ofpp.NXFlowSpecLoad(src=('in_port', 0), dst=('in_port', 0), n_bits=16)
        on_wire = b'\x08\x10\x80\x00\x00\x04\x00\x00\x80\x00\x00\x04\x00\x00'
        self.assertEqual(on_wire, user.serialize())
        o, rest = ofpp._NXFlowSpec.parse(on_wire)
        self.assertEqual(user.to_jsondict(), o.to_jsondict())
        self.assertEqual(str(user), str(o))
        self.assertEqual(b'', rest)

    def test_flowspec_src_1_dst_1(self):
        user = ofpp.NXFlowSpecLoad(src=99, dst=('in_port', 0), n_bits=16)
        on_wire = b'(\x10\x00c\x80\x00\x00\x04\x00\x00'
        self.assertEqual(on_wire, user.serialize())
        o, rest = ofpp._NXFlowSpec.parse(on_wire)
        self.assertEqual(user.to_jsondict(), o.to_jsondict())
        self.assertEqual(str(user), str(o))
        self.assertEqual(b'', rest)

    def test_flowspec_src_0_dst_2(self):
        user = ofpp.NXFlowSpecOutput(src=('in_port', 0), dst='', n_bits=16)
        on_wire = b'\x10\x10\x80\x00\x00\x04\x00\x00'
        self.assertEqual(on_wire, user.serialize())
        o, rest = ofpp._NXFlowSpec.parse(on_wire)
        self.assertEqual(user.to_jsondict(), o.to_jsondict())
        self.assertEqual(str(user), str(o))
        self.assertEqual(b'', rest)