import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
class Test_OXM(unittest.TestCase):

    def _test_encode(self, user, on_wire):
        f, uv = user
        n, v, m = ofp.oxm_from_user(f, uv)
        buf = bytearray()
        ofp.oxm_serialize(n, v, m, buf, 0)
        self.assertEqual(on_wire, buf)

    def _test_decode(self, user, on_wire):
        n, v, m, l = ofp.oxm_parse(on_wire, 0)
        self.assertEqual(len(on_wire), l)
        f, uv = ofp.oxm_to_user(n, v, m)
        self.assertEqual(user, (f, uv))

    def _test_encode_header(self, user, on_wire):
        f = user
        n = ofp.oxm_from_user_header(f)
        buf = bytearray()
        ofp.oxm_serialize_header(n, buf, 0)
        self.assertEqual(on_wire, buf)

    def _test_decode_header(self, user, on_wire):
        n, l = ofp.oxm_parse_header(on_wire, 0)
        self.assertEqual(len(on_wire), l)
        f = ofp.oxm_to_user_header(n)
        self.assertEqual(user, f)

    def _test(self, user, on_wire, header_bytes):
        self._test_encode(user, on_wire)
        self._test_decode(user, on_wire)
        if isinstance(user[1], tuple):
            return
        user_header = user[0]
        on_wire_header = on_wire[:header_bytes]
        self._test_decode_header(user_header, on_wire_header)
        if user_header.startswith('field_'):
            return
        self._test_encode_header(user_header, on_wire_header)

    def test_basic_nomask(self):
        user = ('ipv4_src', '192.0.2.1')
        on_wire = b'\x80\x00\x16\x04\xc0\x00\x02\x01'
        self._test(user, on_wire, 4)

    def test_basic_mask(self):
        user = ('ipv4_src', ('192.0.2.1', '255.255.0.0'))
        on_wire = b'\x80\x00\x17\x08\xc0\x00\x02\x01\xff\xff\x00\x00'
        self._test(user, on_wire, 4)

    def test_exp_nomask(self):
        user = ('_dp_hash', 305419896)
        on_wire = b'\xff\xff\x00\x08\x00\x00# \x124Vx'
        self._test(user, on_wire, 8)

    def test_exp_mask(self):
        user = ('_dp_hash', (305419896, 2147483647))
        on_wire = b'\xff\xff\x01\x0c\x00\x00# \x124Vx\x7f\xff\xff\xff'
        self._test(user, on_wire, 8)

    def test_exp_nomask_2(self):
        user = ('tcp_flags', 2166)
        on_wire = b'\xff\xffT\x06ONF\x00\x08v'
        self._test(user, on_wire, 8)

    def test_exp_mask_2(self):
        user = ('tcp_flags', (2166, 2047))
        on_wire = b'\xff\xffU\x08ONF\x00\x08v\x07\xff'
        self._test(user, on_wire, 8)

    def test_exp_nomask_3(self):
        user = ('actset_output', 2557891634)
        on_wire = b'\xff\xffV\x08ONF\x00\x98vT2'
        self._test(user, on_wire, 8)

    def test_exp_mask_3(self):
        user = ('actset_output', (2557891634, 4294967294))
        on_wire = b'\xff\xffW\x0cONF\x00\x98vT2\xff\xff\xff\xfe'
        self._test(user, on_wire, 8)

    def test_nxm_1_nomask(self):
        user = ('tun_ipv4_src', '192.0.2.1')
        on_wire = b'\x00\x01>\x04\xc0\x00\x02\x01'
        self._test(user, on_wire, 4)

    def test_nxm_1_mask(self):
        user = ('tun_ipv4_src', ('192.0.2.1', '255.255.0.0'))
        on_wire = b'\x00\x01?\x08\xc0\x00\x02\x01\xff\xff\x00\x00'
        self._test(user, on_wire, 4)

    def test_ext_256_nomask(self):
        user = ('pbb_uca', 50)
        on_wire = b'\xff\xff\x00\x07ONF\x00\n\x002'
        self._test(user, on_wire, 10)

    def test_ext_256_mask(self):
        user = ('pbb_uca', (50, 51))
        on_wire = b'\xff\xff\x01\x08ONF\x00\n\x0023'
        self._test(user, on_wire, 10)

    def test_basic_unknown_nomask(self):
        user = ('field_100', 'aG9nZWhvZ2U=')
        on_wire = b'\x00\x00\xc8\x08hogehoge'
        self._test(user, on_wire, 4)

    def test_basic_unknown_mask(self):
        user = ('field_100', ('aG9nZWhvZ2U=', 'ZnVnYWZ1Z2E='))
        on_wire = b'\x00\x00\xc9\x10hogehogefugafuga'
        self._test(user, on_wire, 4)