import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _oid_len_valid(self):
    return self._OID_LEN_MIN <= self.oid_len <= self._OID_LEN_MAX