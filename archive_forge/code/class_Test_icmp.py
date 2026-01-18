import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
class Test_icmp(unittest.TestCase):
    echo_id = None
    echo_seq = None
    echo_data = None
    unreach_mtu = None
    unreach_data = None
    unreach_data_len = None
    te_data = None
    te_data_len = None

    def setUp(self):
        self.type_ = icmp.ICMP_ECHO_REQUEST
        self.code = 0
        self.csum = 0
        self.data = b''
        self.ic = icmp.icmp(self.type_, self.code, self.csum, self.data)
        self.buf = bytearray(struct.pack(icmp.icmp._PACK_STR, self.type_, self.code, self.csum))
        self.csum_calc = packet_utils.checksum(self.buf)
        struct.pack_into('!H', self.buf, 2, self.csum_calc)

    def setUp_with_echo(self):
        self.echo_id = 13379
        self.echo_seq = 1
        self.echo_data = b'0\x0e\t\x00\x00\x00\x00\x00' + b'\x10\x11\x12\x13\x14\x15\x16\x17' + b'\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f' + b' !"#$%&\'' + b'()*+,-./' + b'01234567'
        self.data = icmp.echo(id_=self.echo_id, seq=self.echo_seq, data=self.echo_data)
        self.type_ = icmp.ICMP_ECHO_REQUEST
        self.code = 0
        self.ic = icmp.icmp(self.type_, self.code, self.csum, self.data)
        self.buf = bytearray(struct.pack(icmp.icmp._PACK_STR, self.type_, self.code, self.csum))
        self.buf += self.data.serialize()
        self.csum_calc = packet_utils.checksum(self.buf)
        struct.pack_into('!H', self.buf, 2, self.csum_calc)

    def setUp_with_dest_unreach(self):
        self.unreach_mtu = 10
        self.unreach_data = b'abc'
        self.unreach_data_len = len(self.unreach_data)
        self.data = icmp.dest_unreach(data_len=self.unreach_data_len, mtu=self.unreach_mtu, data=self.unreach_data)
        self.type_ = icmp.ICMP_DEST_UNREACH
        self.code = icmp.ICMP_HOST_UNREACH_CODE
        self.ic = icmp.icmp(self.type_, self.code, self.csum, self.data)
        self.buf = bytearray(struct.pack(icmp.icmp._PACK_STR, self.type_, self.code, self.csum))
        self.buf += self.data.serialize()
        self.csum_calc = packet_utils.checksum(self.buf)
        struct.pack_into('!H', self.buf, 2, self.csum_calc)

    def setUp_with_TimeExceeded(self):
        self.te_data = b'abc'
        self.te_data_len = len(self.te_data)
        self.data = icmp.TimeExceeded(data_len=self.te_data_len, data=self.te_data)
        self.type_ = icmp.ICMP_TIME_EXCEEDED
        self.code = 0
        self.ic = icmp.icmp(self.type_, self.code, self.csum, self.data)
        self.buf = bytearray(struct.pack(icmp.icmp._PACK_STR, self.type_, self.code, self.csum))
        self.buf += self.data.serialize()
        self.csum_calc = packet_utils.checksum(self.buf)
        struct.pack_into('!H', self.buf, 2, self.csum_calc)

    def test_init(self):
        self.assertEqual(self.type_, self.ic.type)
        self.assertEqual(self.code, self.ic.code)
        self.assertEqual(self.csum, self.ic.csum)
        self.assertEqual(str(self.data), str(self.ic.data))

    def test_init_with_echo(self):
        self.setUp_with_echo()
        self.test_init()

    def test_init_with_dest_unreach(self):
        self.setUp_with_dest_unreach()
        self.test_init()

    def test_init_with_TimeExceeded(self):
        self.setUp_with_TimeExceeded()
        self.test_init()

    def test_parser(self):
        _res = icmp.icmp.parser(bytes(self.buf))
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.type_, res.type)
        self.assertEqual(self.code, res.code)
        self.assertEqual(self.csum_calc, res.csum)
        self.assertEqual(str(self.data), str(res.data))

    def test_parser_with_echo(self):
        self.setUp_with_echo()
        self.test_parser()

    def test_parser_with_dest_unreach(self):
        self.setUp_with_dest_unreach()
        self.test_parser()

    def test_parser_with_TimeExceeded(self):
        self.setUp_with_TimeExceeded()
        self.test_parser()

    def test_serialize(self):
        data = bytearray()
        prev = None
        buf = self.ic.serialize(data, prev)
        res = struct.unpack_from(icmp.icmp._PACK_STR, bytes(buf))
        self.assertEqual(self.type_, res[0])
        self.assertEqual(self.code, res[1])
        self.assertEqual(self.csum_calc, res[2])

    def test_serialize_with_echo(self):
        self.setUp_with_echo()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ic.serialize(data, prev)
        echo = icmp.echo.parser(bytes(buf), icmp.icmp._MIN_LEN)
        self.assertEqual(repr(self.data), repr(echo))

    def test_serialize_with_dest_unreach(self):
        self.setUp_with_dest_unreach()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ic.serialize(data, prev)
        unreach = icmp.dest_unreach.parser(bytes(buf), icmp.icmp._MIN_LEN)
        self.assertEqual(repr(self.data), repr(unreach))

    def test_serialize_with_TimeExceeded(self):
        self.setUp_with_TimeExceeded()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ic.serialize(data, prev)
        te = icmp.TimeExceeded.parser(bytes(buf), icmp.icmp._MIN_LEN)
        self.assertEqual(repr(self.data), repr(te))

    def test_to_string(self):
        icmp_values = {'type': repr(self.type_), 'code': repr(self.code), 'csum': repr(self.csum), 'data': repr(self.data)}
        _ic_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, v in inspect.getmembers(self.ic) if k in icmp_values])
        ic_str = '%s(%s)' % (icmp.icmp.__name__, _ic_str)
        self.assertEqual(str(self.ic), ic_str)
        self.assertEqual(repr(self.ic), ic_str)

    def test_to_string_with_echo(self):
        self.setUp_with_echo()
        self.test_to_string()

    def test_to_string_with_dest_unreach(self):
        self.setUp_with_dest_unreach()
        self.test_to_string()

    def test_to_string_with_TimeExceeded(self):
        self.setUp_with_TimeExceeded()
        self.test_to_string()

    def test_default_args(self):
        ic = icmp.icmp()
        buf = ic.serialize(bytearray(), None)
        res = struct.unpack(icmp.icmp._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], 8)
        self.assertEqual(res[1], 0)
        self.assertEqual(buf[4:], b'\x00\x00\x00\x00')
        ic = icmp.icmp(type_=icmp.ICMP_DEST_UNREACH, data=icmp.dest_unreach())
        buf = ic.serialize(bytearray(), None)
        res = struct.unpack(icmp.icmp._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], 3)
        self.assertEqual(res[1], 0)
        self.assertEqual(buf[4:], b'\x00\x00\x00\x00')

    def test_json(self):
        jsondict = self.ic.to_jsondict()
        ic = icmp.icmp.from_jsondict(jsondict['icmp'])
        self.assertEqual(str(self.ic), str(ic))

    def test_json_with_echo(self):
        self.setUp_with_echo()
        self.test_json()

    def test_json_with_dest_unreach(self):
        self.setUp_with_dest_unreach()
        self.test_json()

    def test_json_with_TimeExceeded(self):
        self.setUp_with_TimeExceeded()
        self.test_json()