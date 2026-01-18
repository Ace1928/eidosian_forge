import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import lldp
from os_ken.lib import addrconv
class TestLLDPMandatoryTLV(unittest.TestCase):

    def setUp(self):
        self.data = b'\x01\x80\xc2\x00\x00\x0e\x00\x04' + b'\x96\x1f\xa7&\x88\xcc\x02\x07' + b'\x04\x00\x04\x96\x1f\xa7&\x04' + b'\x04\x051/3\x06\x02\x00' + b'x\x00\x00'

    def tearDown(self):
        pass

    def test_get_tlv_type(self):
        buf = b'\x02\x07\x04\x00\x04\x96\x1f\xa7&'
        self.assertEqual(lldp.LLDPBasicTLV.get_type(buf), lldp.LLDP_TLV_CHASSIS_ID)

    def test_parse_without_ethernet(self):
        buf = self.data[ethernet.ethernet._MIN_LEN:]
        lldp_pkt, cls, rest_buf = lldp.lldp.parser(buf)
        self.assertEqual(len(rest_buf), 0)
        tlvs = lldp_pkt.tlvs
        self.assertEqual(tlvs[0].tlv_type, lldp.LLDP_TLV_CHASSIS_ID)
        self.assertEqual(tlvs[0].len, 7)
        self.assertEqual(tlvs[0].subtype, lldp.ChassisID.SUB_MAC_ADDRESS)
        self.assertEqual(tlvs[0].chassis_id, b'\x00\x04\x96\x1f\xa7&')
        self.assertEqual(tlvs[1].tlv_type, lldp.LLDP_TLV_PORT_ID)
        self.assertEqual(tlvs[1].len, 4)
        self.assertEqual(tlvs[1].subtype, lldp.PortID.SUB_INTERFACE_NAME)
        self.assertEqual(tlvs[1].port_id, b'1/3')
        self.assertEqual(tlvs[2].tlv_type, lldp.LLDP_TLV_TTL)
        self.assertEqual(tlvs[2].len, 2)
        self.assertEqual(tlvs[2].ttl, 120)
        self.assertEqual(tlvs[3].tlv_type, lldp.LLDP_TLV_END)

    def test_parse(self):
        buf = self.data
        pkt = packet.Packet(buf)
        i = iter(pkt)
        self.assertEqual(type(next(i)), ethernet.ethernet)
        self.assertEqual(type(next(i)), lldp.lldp)

    def test_tlv(self):
        tlv = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x04\x96\x1f\xa7&')
        self.assertEqual(tlv.tlv_type, lldp.LLDP_TLV_CHASSIS_ID)
        self.assertEqual(tlv.len, 7)
        typelen, = struct.unpack('!H', b'\x02\x07')
        self.assertEqual(tlv.typelen, typelen)

    def test_serialize_without_ethernet(self):
        tlv_chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x04\x96\x1f\xa7&')
        tlv_port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/3')
        tlv_ttl = lldp.TTL(ttl=120)
        tlv_end = lldp.End()
        tlvs = (tlv_chassis_id, tlv_port_id, tlv_ttl, tlv_end)
        lldp_pkt = lldp.lldp(tlvs)
        self.assertEqual(lldp_pkt.serialize(None, None), self.data[ethernet.ethernet._MIN_LEN:])

    def test_serialize(self):
        pkt = packet.Packet()
        dst = lldp.LLDP_MAC_NEAREST_BRIDGE
        src = '00:04:96:1f:a7:26'
        ethertype = ether.ETH_TYPE_LLDP
        eth_pkt = ethernet.ethernet(dst, src, ethertype)
        pkt.add_protocol(eth_pkt)
        tlv_chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=addrconv.mac.text_to_bin(src))
        tlv_port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/3')
        tlv_ttl = lldp.TTL(ttl=120)
        tlv_end = lldp.End()
        tlvs = (tlv_chassis_id, tlv_port_id, tlv_ttl, tlv_end)
        lldp_pkt = lldp.lldp(tlvs)
        pkt.add_protocol(lldp_pkt)
        self.assertEqual(len(pkt.protocols), 2)
        pkt.serialize()
        data_len = len(self.data)
        pkt_data_lldp = pkt.data[:data_len]
        pkt_data_pad = pkt.data[data_len:]
        self.assertEqual(b'\x00' * (60 - data_len), pkt_data_pad)
        self.assertEqual(self.data, pkt_data_lldp)

    def test_to_string(self):
        chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x04\x96\x1f\xa7&')
        port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/3')
        ttl = lldp.TTL(ttl=120)
        end = lldp.End()
        tlvs = (chassis_id, port_id, ttl, end)
        lldp_pkt = lldp.lldp(tlvs)
        chassis_id_values = {'subtype': lldp.ChassisID.SUB_MAC_ADDRESS, 'chassis_id': b'\x00\x04\x96\x1f\xa7&', 'len': chassis_id.len, 'typelen': chassis_id.typelen}
        _ch_id_str = ','.join(['%s=%s' % (k, repr(chassis_id_values[k])) for k, v in inspect.getmembers(chassis_id) if k in chassis_id_values])
        tlv_chassis_id_str = '%s(%s)' % (lldp.ChassisID.__name__, _ch_id_str)
        port_id_values = {'subtype': port_id.subtype, 'port_id': port_id.port_id, 'len': port_id.len, 'typelen': port_id.typelen}
        _port_id_str = ','.join(['%s=%s' % (k, repr(port_id_values[k])) for k, v in inspect.getmembers(port_id) if k in port_id_values])
        tlv_port_id_str = '%s(%s)' % (lldp.PortID.__name__, _port_id_str)
        ttl_values = {'ttl': ttl.ttl, 'len': ttl.len, 'typelen': ttl.typelen}
        _ttl_str = ','.join(['%s=%s' % (k, repr(ttl_values[k])) for k, v in inspect.getmembers(ttl) if k in ttl_values])
        tlv_ttl_str = '%s(%s)' % (lldp.TTL.__name__, _ttl_str)
        end_values = {'len': end.len, 'typelen': end.typelen}
        _end_str = ','.join(['%s=%s' % (k, repr(end_values[k])) for k, v in inspect.getmembers(end) if k in end_values])
        tlv_end_str = '%s(%s)' % (lldp.End.__name__, _end_str)
        _tlvs_str = '(%s, %s, %s, %s)'
        tlvs_str = _tlvs_str % (tlv_chassis_id_str, tlv_port_id_str, tlv_ttl_str, tlv_end_str)
        _lldp_str = '%s(tlvs=%s)'
        lldp_str = _lldp_str % (lldp.lldp.__name__, tlvs_str)
        self.assertEqual(str(lldp_pkt), lldp_str)
        self.assertEqual(repr(lldp_pkt), lldp_str)

    def test_json(self):
        chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x04\x96\x1f\xa7&')
        port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/3')
        ttl = lldp.TTL(ttl=120)
        end = lldp.End()
        tlvs = (chassis_id, port_id, ttl, end)
        lldp1 = lldp.lldp(tlvs)
        jsondict = lldp1.to_jsondict()
        lldp2 = lldp.lldp.from_jsondict(jsondict['lldp'])
        self.assertEqual(str(lldp1), str(lldp2))