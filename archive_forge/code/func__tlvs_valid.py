import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _tlvs_valid(self):
    return self.tlvs[0].tlv_type == LLDP_TLV_CHASSIS_ID and self.tlvs[1].tlv_type == LLDP_TLV_PORT_ID and (self.tlvs[2].tlv_type == LLDP_TLV_TTL) and (self.tlvs[-1].tlv_type == LLDP_TLV_END)