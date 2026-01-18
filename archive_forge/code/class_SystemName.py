import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@lldp.set_tlv_type(LLDP_TLV_SYSTEM_NAME)
class SystemName(LLDPBasicTLV):
    """System name TLV encoder/decoder class

    ================= =====================================
    Attribute         Description
    ================= =====================================
    buf               Binary data to parse.
    system_name       System name.
    ================= =====================================
    """
    _LEN_MAX = 255

    def __init__(self, buf=None, *args, **kwargs):
        super(SystemName, self).__init__(buf, *args, **kwargs)
        if buf:
            pass
        else:
            self.system_name = kwargs['system_name']
            self.len = len(self.system_name)
            assert self._len_valid()
            self.typelen = self.tlv_type << LLDP_TLV_TYPE_SHIFT | self.len

    def serialize(self):
        return struct.pack('!H', self.typelen) + self.tlv_info

    @property
    def system_name(self):
        return self.tlv_info

    @system_name.setter
    def system_name(self, value):
        self.tlv_info = value