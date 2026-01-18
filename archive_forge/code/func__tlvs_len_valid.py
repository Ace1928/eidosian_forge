import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _tlvs_len_valid(self):
    return len(self.tlvs) >= 4