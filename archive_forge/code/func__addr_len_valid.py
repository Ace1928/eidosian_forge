import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _addr_len_valid(self):
    return self._ADDR_LEN_MIN <= self.addr_len or self.addr_len <= self._ADDR_LEN_MAX