import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_ipv6(unittest.TestCase):

    def setUp(self):
        self.version = 6
        self.traffic_class = 0
        self.flow_label = 0
        self.payload_length = 817
        self.nxt = 6
        self.hop_limit = 128
        self.src = '2002:4637:d5d3::4637:d5d3'
        self.dst = '2001:4860:0:2001::68'
        self.ext_hdrs = []
        self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
        self.v_tc_flow = self.version << 28 | self.traffic_class << 20 | self.flow_label << 12
        self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))

    def setUp_with_hop_opts(self):
        self.opt1_type = 5
        self.opt1_len = 2
        self.opt1_data = b'\x00\x00'
        self.opt2_type = 1
        self.opt2_len = 0
        self.opt2_data = None
        self.options = [ipv6.option(self.opt1_type, self.opt1_len, self.opt1_data), ipv6.option(self.opt2_type, self.opt2_len, self.opt2_data)]
        self.hop_opts_nxt = 6
        self.hop_opts_size = 0
        self.hop_opts = ipv6.hop_opts(self.hop_opts_nxt, self.hop_opts_size, self.options)
        self.ext_hdrs = [self.hop_opts]
        self.payload_length += len(self.hop_opts)
        self.nxt = ipv6.hop_opts.TYPE
        self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
        self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
        self.buf += self.hop_opts.serialize()

    def setUp_with_dst_opts(self):
        self.opt1_type = 5
        self.opt1_len = 2
        self.opt1_data = b'\x00\x00'
        self.opt2_type = 1
        self.opt2_len = 0
        self.opt2_data = None
        self.options = [ipv6.option(self.opt1_type, self.opt1_len, self.opt1_data), ipv6.option(self.opt2_type, self.opt2_len, self.opt2_data)]
        self.dst_opts_nxt = 6
        self.dst_opts_size = 0
        self.dst_opts = ipv6.dst_opts(self.dst_opts_nxt, self.dst_opts_size, self.options)
        self.ext_hdrs = [self.dst_opts]
        self.payload_length += len(self.dst_opts)
        self.nxt = ipv6.dst_opts.TYPE
        self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
        self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
        self.buf += self.dst_opts.serialize()

    def setUp_with_routing_type3(self):
        self.routing_nxt = 6
        self.routing_size = 6
        self.routing_type = 3
        self.routing_seg = 2
        self.routing_cmpi = 0
        self.routing_cmpe = 0
        self.routing_adrs = ['2001:db8:dead::1', '2001:db8:dead::2', '2001:db8:dead::3']
        self.routing = ipv6.routing_type3(self.routing_nxt, self.routing_size, self.routing_type, self.routing_seg, self.routing_cmpi, self.routing_cmpe, self.routing_adrs)
        self.ext_hdrs = [self.routing]
        self.payload_length += len(self.routing)
        self.nxt = ipv6.routing.TYPE
        self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
        self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
        self.buf += self.routing.serialize()

    def setUp_with_fragment(self):
        self.fragment_nxt = 6
        self.fragment_offset = 50
        self.fragment_more = 1
        self.fragment_id = 123
        self.fragment = ipv6.fragment(self.fragment_nxt, self.fragment_offset, self.fragment_more, self.fragment_id)
        self.ext_hdrs = [self.fragment]
        self.payload_length += len(self.fragment)
        self.nxt = ipv6.fragment.TYPE
        self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
        self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
        self.buf += self.fragment.serialize()

    def setUp_with_auth(self):
        self.auth_nxt = 6
        self.auth_size = 4
        self.auth_spi = 256
        self.auth_seq = 1
        self.auth_data = b'\xa0\xe7\xf8\xab\xf9i\x1a\x8b\xf3\x9f|\xae'
        self.auth = ipv6.auth(self.auth_nxt, self.auth_size, self.auth_spi, self.auth_seq, self.auth_data)
        self.ext_hdrs = [self.auth]
        self.payload_length += len(self.auth)
        self.nxt = ipv6.auth.TYPE
        self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
        self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
        self.buf += self.auth.serialize()

    def setUp_with_multi_headers(self):
        self.opt1_type = 5
        self.opt1_len = 2
        self.opt1_data = b'\x00\x00'
        self.opt2_type = 1
        self.opt2_len = 0
        self.opt2_data = None
        self.options = [ipv6.option(self.opt1_type, self.opt1_len, self.opt1_data), ipv6.option(self.opt2_type, self.opt2_len, self.opt2_data)]
        self.hop_opts_nxt = ipv6.auth.TYPE
        self.hop_opts_size = 0
        self.hop_opts = ipv6.hop_opts(self.hop_opts_nxt, self.hop_opts_size, self.options)
        self.auth_nxt = 6
        self.auth_size = 4
        self.auth_spi = 256
        self.auth_seq = 1
        self.auth_data = b'\xa0\xe7\xf8\xab\xf9i\x1a\x8b\xf3\x9f|\xae'
        self.auth = ipv6.auth(self.auth_nxt, self.auth_size, self.auth_spi, self.auth_seq, self.auth_data)
        self.ext_hdrs = [self.hop_opts, self.auth]
        self.payload_length += len(self.hop_opts) + len(self.auth)
        self.nxt = ipv6.hop_opts.TYPE
        self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
        self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
        self.buf += self.hop_opts.serialize()
        self.buf += self.auth.serialize()

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.version, self.ip.version)
        self.assertEqual(self.traffic_class, self.ip.traffic_class)
        self.assertEqual(self.flow_label, self.ip.flow_label)
        self.assertEqual(self.payload_length, self.ip.payload_length)
        self.assertEqual(self.nxt, self.ip.nxt)
        self.assertEqual(self.hop_limit, self.ip.hop_limit)
        self.assertEqual(self.src, self.ip.src)
        self.assertEqual(self.dst, self.ip.dst)
        self.assertEqual(str(self.ext_hdrs), str(self.ip.ext_hdrs))

    def test_init_with_hop_opts(self):
        self.setUp_with_hop_opts()
        self.test_init()

    def test_init_with_dst_opts(self):
        self.setUp_with_dst_opts()
        self.test_init()

    def test_init_with_routing_type3(self):
        self.setUp_with_routing_type3()
        self.test_init()

    def test_init_with_fragment(self):
        self.setUp_with_fragment()
        self.test_init()

    def test_init_with_auth(self):
        self.setUp_with_auth()
        self.test_init()

    def test_init_with_multi_headers(self):
        self.setUp_with_multi_headers()
        self.test_init()

    def test_parser(self):
        _res = self.ip.parser(bytes(self.buf))
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.version, res.version)
        self.assertEqual(self.traffic_class, res.traffic_class)
        self.assertEqual(self.flow_label, res.flow_label)
        self.assertEqual(self.payload_length, res.payload_length)
        self.assertEqual(self.nxt, res.nxt)
        self.assertEqual(self.hop_limit, res.hop_limit)
        self.assertEqual(self.src, res.src)
        self.assertEqual(self.dst, res.dst)
        self.assertEqual(str(self.ext_hdrs), str(res.ext_hdrs))

    def test_parser_with_hop_opts(self):
        self.setUp_with_hop_opts()
        self.test_parser()

    def test_parser_with_dst_opts(self):
        self.setUp_with_dst_opts()
        self.test_parser()

    def test_parser_with_routing_type3(self):
        self.setUp_with_routing_type3()
        self.test_parser()

    def test_parser_with_fragment(self):
        self.setUp_with_fragment()
        self.test_parser()

    def test_parser_with_auth(self):
        self.setUp_with_auth()
        self.test_parser()

    def test_parser_with_multi_headers(self):
        self.setUp_with_multi_headers()
        self.test_parser()

    def test_serialize(self):
        data = bytearray()
        prev = None
        buf = self.ip.serialize(data, prev)
        res = struct.unpack_from(ipv6.ipv6._PACK_STR, bytes(buf))
        self.assertEqual(self.v_tc_flow, res[0])
        self.assertEqual(self.payload_length, res[1])
        self.assertEqual(self.nxt, res[2])
        self.assertEqual(self.hop_limit, res[3])
        self.assertEqual(self.src, addrconv.ipv6.bin_to_text(res[4]))
        self.assertEqual(self.dst, addrconv.ipv6.bin_to_text(res[5]))

    def test_serialize_with_hop_opts(self):
        self.setUp_with_hop_opts()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ip.serialize(data, prev)
        hop_opts = ipv6.hop_opts.parser(bytes(buf[ipv6.ipv6._MIN_LEN:]))
        self.assertEqual(repr(self.hop_opts), repr(hop_opts))

    def test_serialize_with_dst_opts(self):
        self.setUp_with_dst_opts()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ip.serialize(data, prev)
        dst_opts = ipv6.dst_opts.parser(bytes(buf[ipv6.ipv6._MIN_LEN:]))
        self.assertEqual(repr(self.dst_opts), repr(dst_opts))

    def test_serialize_with_routing_type3(self):
        self.setUp_with_routing_type3()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ip.serialize(data, prev)
        routing = ipv6.routing.parser(bytes(buf[ipv6.ipv6._MIN_LEN:]))
        self.assertEqual(repr(self.routing), repr(routing))

    def test_serialize_with_fragment(self):
        self.setUp_with_fragment()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ip.serialize(data, prev)
        fragment = ipv6.fragment.parser(bytes(buf[ipv6.ipv6._MIN_LEN:]))
        self.assertEqual(repr(self.fragment), repr(fragment))

    def test_serialize_with_auth(self):
        self.setUp_with_auth()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ip.serialize(data, prev)
        auth = ipv6.auth.parser(bytes(buf[ipv6.ipv6._MIN_LEN:]))
        self.assertEqual(repr(self.auth), repr(auth))

    def test_serialize_with_multi_headers(self):
        self.setUp_with_multi_headers()
        self.test_serialize()
        data = bytearray()
        prev = None
        buf = self.ip.serialize(data, prev)
        offset = ipv6.ipv6._MIN_LEN
        hop_opts = ipv6.hop_opts.parser(bytes(buf[offset:]))
        offset += len(hop_opts)
        auth = ipv6.auth.parser(bytes(buf[offset:]))
        self.assertEqual(repr(self.hop_opts), repr(hop_opts))
        self.assertEqual(repr(self.auth), repr(auth))

    def test_to_string(self):
        ipv6_values = {'version': self.version, 'traffic_class': self.traffic_class, 'flow_label': self.flow_label, 'payload_length': self.payload_length, 'nxt': self.nxt, 'hop_limit': self.hop_limit, 'src': repr(self.src), 'dst': repr(self.dst), 'ext_hdrs': self.ext_hdrs}
        _ipv6_str = ','.join(['%s=%s' % (k, ipv6_values[k]) for k, v in inspect.getmembers(self.ip) if k in ipv6_values])
        ipv6_str = '%s(%s)' % (ipv6.ipv6.__name__, _ipv6_str)
        self.assertEqual(str(self.ip), ipv6_str)
        self.assertEqual(repr(self.ip), ipv6_str)

    def test_to_string_with_hop_opts(self):
        self.setUp_with_hop_opts()
        self.test_to_string()

    def test_to_string_with_dst_opts(self):
        self.setUp_with_dst_opts()
        self.test_to_string()

    def test_to_string_with_fragment(self):
        self.setUp_with_fragment()
        self.test_to_string()

    def test_to_string_with_auth(self):
        self.setUp_with_auth()
        self.test_to_string()

    def test_to_string_with_multi_headers(self):
        self.setUp_with_multi_headers()
        self.test_to_string()

    def test_len(self):
        self.assertEqual(len(self.ip), 40)

    def test_len_with_hop_opts(self):
        self.setUp_with_hop_opts()
        self.assertEqual(len(self.ip), 40 + len(self.hop_opts))

    def test_len_with_dst_opts(self):
        self.setUp_with_dst_opts()
        self.assertEqual(len(self.ip), 40 + len(self.dst_opts))

    def test_len_with_routing_type3(self):
        self.setUp_with_routing_type3()
        self.assertEqual(len(self.ip), 40 + len(self.routing))

    def test_len_with_fragment(self):
        self.setUp_with_fragment()
        self.assertEqual(len(self.ip), 40 + len(self.fragment))

    def test_len_with_auth(self):
        self.setUp_with_auth()
        self.assertEqual(len(self.ip), 40 + len(self.auth))

    def test_len_with_multi_headers(self):
        self.setUp_with_multi_headers()
        self.assertEqual(len(self.ip), 40 + len(self.hop_opts) + len(self.auth))

    def test_default_args(self):
        ip = ipv6.ipv6()
        buf = ip.serialize(bytearray(), None)
        res = struct.unpack(ipv6.ipv6._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 6 << 28)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 6)
        self.assertEqual(res[3], 255)
        self.assertEqual(res[4], addrconv.ipv6.text_to_bin('10::10'))
        self.assertEqual(res[5], addrconv.ipv6.text_to_bin('20::20'))
        ip = ipv6.ipv6(nxt=0, ext_hdrs=[ipv6.hop_opts(58, 0, [ipv6.option(5, 2, b'\x00\x00'), ipv6.option(1, 0, None)])])
        buf = ip.serialize(bytearray(), None)
        res = struct.unpack(ipv6.ipv6._PACK_STR + '8s', bytes(buf))
        self.assertEqual(res[0], 6 << 28)
        self.assertEqual(res[1], 8)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 255)
        self.assertEqual(res[4], addrconv.ipv6.text_to_bin('10::10'))
        self.assertEqual(res[5], addrconv.ipv6.text_to_bin('20::20'))
        self.assertEqual(res[6], b':\x00\x05\x02\x00\x00\x01\x00')

    def test_json(self):
        jsondict = self.ip.to_jsondict()
        ip = ipv6.ipv6.from_jsondict(jsondict['ipv6'])
        self.assertEqual(str(self.ip), str(ip))

    def test_json_with_hop_opts(self):
        self.setUp_with_hop_opts()
        self.test_json()

    def test_json_with_dst_opts(self):
        self.setUp_with_dst_opts()
        self.test_json()

    def test_json_with_routing_type3(self):
        self.setUp_with_routing_type3()
        self.test_json()

    def test_json_with_fragment(self):
        self.setUp_with_fragment()
        self.test_json()

    def test_json_with_auth(self):
        self.setUp_with_auth()
        self.test_json()

    def test_json_with_multi_headers(self):
        self.setUp_with_multi_headers()
        self.test_json()