import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@lldp.set_tlv_type(LLDP_TLV_MANAGEMENT_ADDRESS)
class ManagementAddress(LLDPBasicTLV):
    """Management Address TLV encoder/decoder class

    ================= =====================================
    Attribute         Description
    ================= =====================================
    buf               Binary data to parse.
    addr_subtype      Address type.
    addr              Device address.
    intf_subtype      Interface type.
    intf_num          Interface number.
    oid               Object ID.
    ================= =====================================
    """
    _LEN_MIN = 9
    _LEN_MAX = 167
    _ADDR_PACK_STR = '!BB'
    _ADDR_PACK_SIZE = struct.calcsize(_ADDR_PACK_STR)
    _ADDR_LEN_MIN = 1
    _ADDR_LEN_MAX = 31
    _INTF_PACK_STR = '!BIB'
    _INTF_PACK_SIZE = struct.calcsize(_INTF_PACK_STR)
    _OID_LEN_MIN = 0
    _OID_LEN_MAX = 128

    def __init__(self, buf=None, *args, **kwargs):
        super(ManagementAddress, self).__init__(buf, *args, **kwargs)
        if buf:
            self.addr_len, self.addr_subtype = struct.unpack(self._ADDR_PACK_STR, self.tlv_info[:self._ADDR_PACK_SIZE])
            assert self._addr_len_valid()
            offset = self._ADDR_PACK_SIZE + self.addr_len - 1
            self.addr = self.tlv_info[self._ADDR_PACK_SIZE:offset]
            self.intf_subtype, self.intf_num, self.oid_len = struct.unpack(self._INTF_PACK_STR, self.tlv_info[offset:offset + self._INTF_PACK_SIZE])
            assert self._oid_len_valid()
            offset = offset + self._INTF_PACK_SIZE
            self.oid = self.tlv_info[offset:]
        else:
            self.addr_subtype = kwargs['addr_subtype']
            self.addr = kwargs['addr']
            self.addr_len = len(self.addr) + 1
            assert self._addr_len_valid()
            self.intf_subtype = kwargs['intf_subtype']
            self.intf_num = kwargs['intf_num']
            self.oid = kwargs['oid']
            self.oid_len = len(self.oid)
            assert self._oid_len_valid()
            self.len = self._ADDR_PACK_SIZE + self.addr_len - 1 + self._INTF_PACK_SIZE + self.oid_len
            assert self._len_valid()
            self.typelen = self.tlv_type << LLDP_TLV_TYPE_SHIFT | self.len

    def serialize(self):
        tlv_info = struct.pack(self._ADDR_PACK_STR, self.addr_len, self.addr_subtype)
        tlv_info += self.addr
        tlv_info += struct.pack(self._INTF_PACK_STR, self.intf_subtype, self.intf_num, self.oid_len)
        tlv_info += self.oid
        return struct.pack('!H', self.typelen) + tlv_info

    def _addr_len_valid(self):
        return self._ADDR_LEN_MIN <= self.addr_len or self.addr_len <= self._ADDR_LEN_MAX

    def _oid_len_valid(self):
        return self._OID_LEN_MIN <= self.oid_len <= self._OID_LEN_MAX