import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _len_valid(self):
    return self._LEN_MIN <= self.len and self.len <= self._LEN_MAX