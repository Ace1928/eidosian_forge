import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@staticmethod
def _decode_bridge_id(bridge_id):
    priority = bridge_id >> 48 & 61440
    system_id_extension = bridge_id >> 48 & 4095
    mac_addr = bridge_id & 281474976710655
    mac_addr_list = [format(mac_addr >> 8 * i & 255, '02x') for i in range(0, 6)]
    mac_addr_list.reverse()
    mac_address_bin = binascii.a2b_hex(''.join(mac_addr_list))
    mac_address = addrconv.mac.bin_to_text(mac_address_bin)
    return (priority, system_id_extension, mac_address)