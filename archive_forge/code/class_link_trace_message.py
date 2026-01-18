import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@cfm.register_cfm_opcode(CFM_LINK_TRACE_MESSAGE)
class link_trace_message(link_trace):
    """CFM (IEEE Std 802.1ag-2007) Linktrace Message (LTM)
    encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ==================== =======================================
    Attribute            Description
    ==================== =======================================
    md_lv                Maintenance Domain Level.
    version              The protocol version number.
    use_fdb_only         UseFDBonly bit.
    transaction_id       LTM Transaction Identifier.
    ttl                  LTM TTL.
    ltm_orig_addr        Original MAC Address.
    ltm_targ_addr        Target MAC Address.
    tlvs                 TLVs.
    ==================== =======================================
    """
    _PACK_STR = '!4BIB6s6s'
    _ALL_PACK_LEN = struct.calcsize(_PACK_STR)
    _MIN_LEN = _ALL_PACK_LEN
    _TLV_OFFSET = 17
    _TYPE = {'ascii': ['ltm_orig_addr', 'ltm_targ_addr']}

    def __init__(self, md_lv=0, version=CFM_VERSION, use_fdb_only=1, transaction_id=0, ttl=64, ltm_orig_addr='00:00:00:00:00:00', ltm_targ_addr='00:00:00:00:00:00', tlvs=None):
        super(link_trace_message, self).__init__(md_lv, version, use_fdb_only, transaction_id, ttl, tlvs)
        self._opcode = CFM_LINK_TRACE_MESSAGE
        self.ltm_orig_addr = ltm_orig_addr
        self.ltm_targ_addr = ltm_targ_addr

    @classmethod
    def parser(cls, buf):
        md_lv_version, opcode, flags, tlv_offset, transaction_id, ttl, ltm_orig_addr, ltm_targ_addr = struct.unpack_from(cls._PACK_STR, buf)
        md_lv = int(md_lv_version >> 5)
        version = int(md_lv_version & 31)
        use_fdb_only = int(flags >> 7)
        tlvs = cls._parser_tlvs(buf[cls._MIN_LEN:])
        return cls(md_lv, version, use_fdb_only, transaction_id, ttl, addrconv.mac.bin_to_text(ltm_orig_addr), addrconv.mac.bin_to_text(ltm_targ_addr), tlvs)

    def serialize(self):
        buf = struct.pack(self._PACK_STR, self.md_lv << 5 | self.version, self._opcode, self.use_fdb_only << 7, self._TLV_OFFSET, self.transaction_id, self.ttl, addrconv.mac.text_to_bin(self.ltm_orig_addr), addrconv.mac.text_to_bin(self.ltm_targ_addr))
        buf = bytearray(buf)
        if self.tlvs:
            buf.extend(self._serialize_tlvs(self.tlvs))
        buf.extend(struct.pack('!B', CFM_END_TLV))
        return buf