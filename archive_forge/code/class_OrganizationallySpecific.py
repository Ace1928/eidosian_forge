import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@lldp.set_tlv_type(LLDP_TLV_ORGANIZATIONALLY_SPECIFIC)
class OrganizationallySpecific(LLDPBasicTLV):
    """Organizationally Specific TLV encoder/decoder class

    ================= =============================================
    Attribute         Description
    ================= =============================================
    buf               Binary data to parse.
    oui               Organizationally unique ID.
    subtype           Organizationally defined subtype.
    info              Organizationally defined information string.
    ================= =============================================
    """
    _PACK_STR = '!3sB'
    _PACK_SIZE = struct.calcsize(_PACK_STR)
    _LEN_MIN = _PACK_SIZE
    _LEN_MAX = 511

    def __init__(self, buf=None, *args, **kwargs):
        super(OrganizationallySpecific, self).__init__(buf, *args, **kwargs)
        if buf:
            self.oui, self.subtype = struct.unpack(self._PACK_STR, self.tlv_info[:self._PACK_SIZE])
            self.info = self.tlv_info[self._PACK_SIZE:]
        else:
            self.oui = kwargs['oui']
            self.subtype = kwargs['subtype']
            self.info = kwargs['info']
            self.len = self._PACK_SIZE + len(self.info)
            assert self._len_valid()
            self.typelen = self.tlv_type << LLDP_TLV_TYPE_SHIFT | self.len

    def serialize(self):
        return struct.pack('!H3sB', self.typelen, self.oui, self.subtype) + self.info