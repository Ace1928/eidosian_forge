import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@cfm.register_cfm_opcode(CFM_LINK_TRACE_REPLY)
class link_trace_reply(link_trace):
    """CFM (IEEE Std 802.1ag-2007) Linktrace Reply (LTR) encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ==================== =======================================
    Attribute            Description
    ==================== =======================================
    version              The protocol version number.
    use_fdb_only         UseFDBonly bit.
    fwd_yes              FwdYes bit.
    terminal_mep         TerminalMep bit.
    transaction_id       LTR Transaction Identifier.
    ttl                  Reply TTL.
    relay_action         Relay Action.The default is 1 (RlyHit)
    tlvs                 TLVs.
    ==================== =======================================
    """
    _PACK_STR = '!4BIBB'
    _ALL_PACK_LEN = struct.calcsize(_PACK_STR)
    _MIN_LEN = _ALL_PACK_LEN
    _TLV_OFFSET = 6
    _RLY_HIT = 1
    _RLY_FDB = 2
    _RLY_MPDB = 3

    def __init__(self, md_lv=0, version=CFM_VERSION, use_fdb_only=1, fwd_yes=0, terminal_mep=1, transaction_id=0, ttl=64, relay_action=_RLY_HIT, tlvs=None):
        super(link_trace_reply, self).__init__(md_lv, version, use_fdb_only, transaction_id, ttl, tlvs)
        self._opcode = CFM_LINK_TRACE_REPLY
        assert fwd_yes in [0, 1]
        self.fwd_yes = fwd_yes
        assert terminal_mep in [0, 1]
        self.terminal_mep = terminal_mep
        assert relay_action in [self._RLY_HIT, self._RLY_FDB, self._RLY_MPDB]
        self.relay_action = relay_action

    @classmethod
    def parser(cls, buf):
        md_lv_version, opcode, flags, tlv_offset, transaction_id, ttl, relay_action = struct.unpack_from(cls._PACK_STR, buf)
        md_lv = int(md_lv_version >> 5)
        version = int(md_lv_version & 31)
        use_fdb_only = int(flags >> 7)
        fwd_yes = int(flags >> 6 & 1)
        terminal_mep = int(flags >> 5 & 1)
        tlvs = cls._parser_tlvs(buf[cls._MIN_LEN:])
        return cls(md_lv, version, use_fdb_only, fwd_yes, terminal_mep, transaction_id, ttl, relay_action, tlvs)

    def serialize(self):
        buf = struct.pack(self._PACK_STR, self.md_lv << 5 | self.version, self._opcode, self.use_fdb_only << 7 | self.fwd_yes << 6 | self.terminal_mep << 5, self._TLV_OFFSET, self.transaction_id, self.ttl, self.relay_action)
        buf = bytearray(buf)
        if self.tlvs:
            buf.extend(self._serialize_tlvs(self.tlvs))
        buf.extend(struct.pack('!B', CFM_END_TLV))
        return buf