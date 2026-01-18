import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import lldp
from os_ken.lib import addrconv
class TestLLDPOptionalTLV(unittest.TestCase):

    def setUp(self):
        self.data = b'\x01\x80\xc2\x00\x00\x0e\x00\x01' + b'0\xf9\xad\xa0\x88\xcc\x02\x07' + b'\x04\x00\x010\xf9\xad\xa0\x04' + b'\x04\x051/1\x06\x02\x00' + b'x\x08\x17Summi' + b't300-48-' + b'Port 100' + b'1\x00\n\rSumm' + b'it300-48' + b'\x00\x0cLSummi' + b't300-48 ' + b'- Versio' + b'n 7.4e.1' + b' (Build ' + b'5) by Re' + b'lease_Ma' + b'ster 05/' + b'27/05 04' + b':53:11\x00\x0e' + b'\x04\x00\x14\x00\x14\x10\x0e\x07' + b'\x06\x00\x010\xf9\xad\xa0\x02' + b'\x00\x00\x03\xe9\x00\xfe\x07\x00' + b'\x12\x0f\x02\x07\x01\x00\xfe\t' + b'\x00\x12\x0f\x01\x03l\x00\x00' + b'\x10\xfe\t\x00\x12\x0f\x03\x01' + b'\x00\x00\x00\x00\xfe\x06\x00\x12' + b'\x0f\x04\x05\xf2\xfe\x06\x00\x80' + b'\xc2\x01\x01\xe8\xfe\x07\x00\x80' + b'\xc2\x02\x01\x00\x00\xfe\x17\x00' + b'\x80\xc2\x03\x01\xe8\x10v2' + b'-0488-03' + b'-0505\x00\xfe\x05' + b'\x00\x80\xc2\x04\x00\x00\x00'

    def tearDown(self):
        pass

    def test_parse(self):
        buf = self.data
        pkt = packet.Packet(buf)
        i = iter(pkt)
        self.assertEqual(type(next(i)), ethernet.ethernet)
        lldp_pkt = next(i)
        self.assertEqual(type(lldp_pkt), lldp.lldp)
        tlvs = lldp_pkt.tlvs
        self.assertEqual(tlvs[3].tlv_type, lldp.LLDP_TLV_PORT_DESCRIPTION)
        self.assertEqual(tlvs[3].port_description, b'Summit300-48-Port 1001\x00')
        self.assertEqual(tlvs[4].tlv_type, lldp.LLDP_TLV_SYSTEM_NAME)
        self.assertEqual(tlvs[4].system_name, b'Summit300-48\x00')
        self.assertEqual(tlvs[5].tlv_type, lldp.LLDP_TLV_SYSTEM_DESCRIPTION)
        self.assertEqual(tlvs[5].system_description, b'Summit300-48 - Version 7.4e.1 (Build 5) ' + b'by Release_Master 05/27/05 04:53:11\x00')
        self.assertEqual(tlvs[6].tlv_type, lldp.LLDP_TLV_SYSTEM_CAPABILITIES)
        self.assertEqual(tlvs[6].system_cap & lldp.SystemCapabilities.CAP_MAC_BRIDGE, lldp.SystemCapabilities.CAP_MAC_BRIDGE)
        self.assertEqual(tlvs[6].enabled_cap & lldp.SystemCapabilities.CAP_MAC_BRIDGE, lldp.SystemCapabilities.CAP_MAC_BRIDGE)
        self.assertEqual(tlvs[6].system_cap & lldp.SystemCapabilities.CAP_TELEPHONE, 0)
        self.assertEqual(tlvs[6].enabled_cap & lldp.SystemCapabilities.CAP_TELEPHONE, 0)
        self.assertEqual(tlvs[7].tlv_type, lldp.LLDP_TLV_MANAGEMENT_ADDRESS)
        self.assertEqual(tlvs[7].addr_len, 7)
        self.assertEqual(tlvs[7].addr, b'\x00\x010\xf9\xad\xa0')
        self.assertEqual(tlvs[7].intf_num, 1001)
        self.assertEqual(tlvs[8].tlv_type, lldp.LLDP_TLV_ORGANIZATIONALLY_SPECIFIC)
        self.assertEqual(tlvs[8].oui, b'\x00\x12\x0f')
        self.assertEqual(tlvs[8].subtype, 2)
        self.assertEqual(tlvs[16].tlv_type, lldp.LLDP_TLV_END)

    def test_parse_corrupted(self):
        buf = self.data
        pkt = packet.Packet(buf[:128])

    def test_serialize(self):
        pkt = packet.Packet()
        dst = lldp.LLDP_MAC_NEAREST_BRIDGE
        src = '00:01:30:f9:ad:a0'
        ethertype = ether.ETH_TYPE_LLDP
        eth_pkt = ethernet.ethernet(dst, src, ethertype)
        pkt.add_protocol(eth_pkt)
        tlv_chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=addrconv.mac.text_to_bin(src))
        tlv_port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/1')
        tlv_ttl = lldp.TTL(ttl=120)
        tlv_port_description = lldp.PortDescription(port_description=b'Summit300-48-Port 1001\x00')
        tlv_system_name = lldp.SystemName(system_name=b'Summit300-48\x00')
        tlv_system_description = lldp.SystemDescription(system_description=b'Summit300-48 - Version 7.4e.1 (Build 5) ' + b'by Release_Master 05/27/05 04:53:11\x00')
        tlv_system_capabilities = lldp.SystemCapabilities(system_cap=20, enabled_cap=20)
        tlv_management_address = lldp.ManagementAddress(addr_subtype=6, addr=b'\x00\x010\xf9\xad\xa0', intf_subtype=2, intf_num=1001, oid=b'')
        tlv_organizationally_specific = lldp.OrganizationallySpecific(oui=b'\x00\x12\x0f', subtype=2, info=b'\x07\x01\x00')
        tlv_end = lldp.End()
        tlvs = (tlv_chassis_id, tlv_port_id, tlv_ttl, tlv_port_description, tlv_system_name, tlv_system_description, tlv_system_capabilities, tlv_management_address, tlv_organizationally_specific, tlv_end)
        lldp_pkt = lldp.lldp(tlvs)
        pkt.add_protocol(lldp_pkt)
        self.assertEqual(len(pkt.protocols), 2)
        pkt.serialize()
        data = bytes(pkt.data[:-2])
        self.assertEqual(data, self.data[:len(data)])

    def test_to_string(self):
        chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x010\xf9\xad\xa0')
        port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/1')
        ttl = lldp.TTL(ttl=120)
        port_desc = lldp.PortDescription(port_description=b'Summit300-48-Port 1001\x00')
        sys_name = lldp.SystemName(system_name=b'Summit300-48\x00')
        sys_desc = lldp.SystemDescription(system_description=b'Summit300-48 - Version 7.4e.1 (Build 5) ' + b'by Release_Master 05/27/05 04:53:11\x00')
        sys_cap = lldp.SystemCapabilities(system_cap=20, enabled_cap=20)
        man_addr = lldp.ManagementAddress(addr_subtype=6, addr=b'\x00\x010\xf9\xad\xa0', intf_subtype=2, intf_num=1001, oid='')
        org_spec = lldp.OrganizationallySpecific(oui=b'\x00\x12\x0f', subtype=2, info=b'\x07\x01\x00')
        end = lldp.End()
        tlvs = (chassis_id, port_id, ttl, port_desc, sys_name, sys_desc, sys_cap, man_addr, org_spec, end)
        lldp_pkt = lldp.lldp(tlvs)
        chassis_id_values = {'subtype': lldp.ChassisID.SUB_MAC_ADDRESS, 'chassis_id': b'\x00\x010\xf9\xad\xa0', 'len': chassis_id.len, 'typelen': chassis_id.typelen}
        _ch_id_str = ','.join(['%s=%s' % (k, repr(chassis_id_values[k])) for k, v in inspect.getmembers(chassis_id) if k in chassis_id_values])
        tlv_chassis_id_str = '%s(%s)' % (lldp.ChassisID.__name__, _ch_id_str)
        port_id_values = {'subtype': port_id.subtype, 'port_id': port_id.port_id, 'len': port_id.len, 'typelen': port_id.typelen}
        _port_id_str = ','.join(['%s=%s' % (k, repr(port_id_values[k])) for k, v in inspect.getmembers(port_id) if k in port_id_values])
        tlv_port_id_str = '%s(%s)' % (lldp.PortID.__name__, _port_id_str)
        ttl_values = {'ttl': ttl.ttl, 'len': ttl.len, 'typelen': ttl.typelen}
        _ttl_str = ','.join(['%s=%s' % (k, repr(ttl_values[k])) for k, v in inspect.getmembers(ttl) if k in ttl_values])
        tlv_ttl_str = '%s(%s)' % (lldp.TTL.__name__, _ttl_str)
        port_desc_values = {'tlv_info': port_desc.tlv_info, 'len': port_desc.len, 'typelen': port_desc.typelen}
        _port_desc_str = ','.join(['%s=%s' % (k, repr(port_desc_values[k])) for k, v in inspect.getmembers(port_desc) if k in port_desc_values])
        tlv_port_desc_str = '%s(%s)' % (lldp.PortDescription.__name__, _port_desc_str)
        sys_name_values = {'tlv_info': sys_name.tlv_info, 'len': sys_name.len, 'typelen': sys_name.typelen}
        _system_name_str = ','.join(['%s=%s' % (k, repr(sys_name_values[k])) for k, v in inspect.getmembers(sys_name) if k in sys_name_values])
        tlv_system_name_str = '%s(%s)' % (lldp.SystemName.__name__, _system_name_str)
        sys_desc_values = {'tlv_info': sys_desc.tlv_info, 'len': sys_desc.len, 'typelen': sys_desc.typelen}
        _sys_desc_str = ','.join(['%s=%s' % (k, repr(sys_desc_values[k])) for k, v in inspect.getmembers(sys_desc) if k in sys_desc_values])
        tlv_sys_desc_str = '%s(%s)' % (lldp.SystemDescription.__name__, _sys_desc_str)
        sys_cap_values = {'system_cap': 20, 'enabled_cap': 20, 'len': sys_cap.len, 'typelen': sys_cap.typelen}
        _sys_cap_str = ','.join(['%s=%s' % (k, repr(sys_cap_values[k])) for k, v in inspect.getmembers(sys_cap) if k in sys_cap_values])
        tlv_sys_cap_str = '%s(%s)' % (lldp.SystemCapabilities.__name__, _sys_cap_str)
        man_addr_values = {'addr_subtype': 6, 'addr': b'\x00\x010\xf9\xad\xa0', 'addr_len': man_addr.addr_len, 'intf_subtype': 2, 'intf_num': 1001, 'oid': '', 'oid_len': man_addr.oid_len, 'len': man_addr.len, 'typelen': man_addr.typelen}
        _man_addr_str = ','.join(['%s=%s' % (k, repr(man_addr_values[k])) for k, v in inspect.getmembers(man_addr) if k in man_addr_values])
        tlv_man_addr_str = '%s(%s)' % (lldp.ManagementAddress.__name__, _man_addr_str)
        org_spec_values = {'oui': b'\x00\x12\x0f', 'subtype': 2, 'info': b'\x07\x01\x00', 'len': org_spec.len, 'typelen': org_spec.typelen}
        _org_spec_str = ','.join(['%s=%s' % (k, repr(org_spec_values[k])) for k, v in inspect.getmembers(org_spec) if k in org_spec_values])
        tlv_org_spec_str = '%s(%s)' % (lldp.OrganizationallySpecific.__name__, _org_spec_str)
        end_values = {'len': end.len, 'typelen': end.typelen}
        _end_str = ','.join(['%s=%s' % (k, repr(end_values[k])) for k, v in inspect.getmembers(end) if k in end_values])
        tlv_end_str = '%s(%s)' % (lldp.End.__name__, _end_str)
        _tlvs_str = '(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
        tlvs_str = _tlvs_str % (tlv_chassis_id_str, tlv_port_id_str, tlv_ttl_str, tlv_port_desc_str, tlv_system_name_str, tlv_sys_desc_str, tlv_sys_cap_str, tlv_man_addr_str, tlv_org_spec_str, tlv_end_str)
        _lldp_str = '%s(tlvs=%s)'
        lldp_str = _lldp_str % (lldp.lldp.__name__, tlvs_str)
        self.assertEqual(str(lldp_pkt), lldp_str)
        self.assertEqual(repr(lldp_pkt), lldp_str)

    def test_json(self):
        chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x010\xf9\xad\xa0')
        port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/1')
        ttl = lldp.TTL(ttl=120)
        port_desc = lldp.PortDescription(port_description=b'Summit300-48-Port 1001\x00')
        sys_name = lldp.SystemName(system_name=b'Summit300-48\x00')
        sys_desc = lldp.SystemDescription(system_description=b'Summit300-48 - Version 7.4e.1 (Build 5) ' + b'by Release_Master 05/27/05 04:53:11\x00')
        sys_cap = lldp.SystemCapabilities(system_cap=20, enabled_cap=20)
        man_addr = lldp.ManagementAddress(addr_subtype=6, addr=b'\x00\x010\xf9\xad\xa0', intf_subtype=2, intf_num=1001, oid='')
        org_spec = lldp.OrganizationallySpecific(oui=b'\x00\x12\x0f', subtype=2, info=b'\x07\x01\x00')
        end = lldp.End()
        tlvs = (chassis_id, port_id, ttl, port_desc, sys_name, sys_desc, sys_cap, man_addr, org_spec, end)
        lldp1 = lldp.lldp(tlvs)
        jsondict = lldp1.to_jsondict()
        lldp2 = lldp.lldp.from_jsondict(jsondict['lldp'])
        self.assertEqual(str(lldp1), str(lldp2))