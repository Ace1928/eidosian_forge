import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@cfm.register_cfm_opcode(CFM_CC_MESSAGE)
class cc_message(operation):
    """CFM (IEEE Std 802.1ag-2007) Continuity Check Message (CCM)
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
    rdi                  RDI bit.
    interval             CCM Interval.The default is 4 (1 frame/s)
    seq_num              Sequence Number.
    mep_id               Maintenance association End Point Identifier.
    md_name_format       Maintenance Domain Name Format.
                         The default is 4 (Character string)
    md_name_length       Maintenance Domain Name Length.
                         (0 means automatically-calculate
                         when encoding.)
    md_name              Maintenance Domain Name.
    short_ma_name_format Short MA Name Format.
                         The default is 2 (Character string)
    short_ma_name_length Short MA Name Format Length.
                         (0 means automatically-calculate
                         when encoding.)
    short_ma_name        Short MA Name.
    tlvs                 TLVs.
    ==================== =======================================
    """
    _PACK_STR = '!4BIHB'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TLV_OFFSET = 70
    _MD_NAME_FORMAT_LEN = 1
    _MD_NAME_LENGTH_LEN = 1
    _SHORT_MA_NAME_FORMAT_LEN = 1
    _SHORT_MA_NAME_LENGTH_LEN = 1
    _MA_ID_LEN = 64
    _MD_FMT_NO_MD_NAME_PRESENT = 1
    _MD_FMT_DOMAIN_NAME_BASED_STRING = 2
    _MD_FMT_MAC_ADDRESS_TWO_OCTET_INTEGER = 3
    _MD_FMT_CHARACTER_STRING = 4
    _SHORT_MA_FMT_PRIMARY_VID = 1
    _SHORT_MA_FMT_CHARACTER_STRING = 2
    _SHORT_MA_FMT_TWO_OCTET_INTEGER = 3
    _SHORT_MA_FMT_RFC_2685_VPN_ID = 4
    _INTERVAL_300_HZ = 1
    _INTERVAL_10_MSEC = 2
    _INTERVAL_100_MSEC = 3
    _INTERVAL_1_SEC = 4
    _INTERVAL_10_SEC = 5
    _INTERVAL_1_MIN = 6
    _INTERVAL_10_MIN = 6

    def __init__(self, md_lv=0, version=CFM_VERSION, rdi=0, interval=_INTERVAL_1_SEC, seq_num=0, mep_id=1, md_name_format=_MD_FMT_CHARACTER_STRING, md_name_length=0, md_name=b'0', short_ma_name_format=_SHORT_MA_FMT_CHARACTER_STRING, short_ma_name_length=0, short_ma_name=b'1', tlvs=None):
        super(cc_message, self).__init__(md_lv, version, tlvs)
        self._opcode = CFM_CC_MESSAGE
        assert rdi in [0, 1]
        self.rdi = rdi
        assert interval != 0
        self.interval = interval
        self.seq_num = seq_num
        assert 1 <= mep_id <= 8191
        self.mep_id = mep_id
        self.md_name_format = md_name_format
        self.md_name_length = md_name_length
        self.md_name = md_name
        self.short_ma_name_format = short_ma_name_format
        self.short_ma_name_length = short_ma_name_length
        self.short_ma_name = short_ma_name

    @classmethod
    def parser(cls, buf):
        md_lv_version, opcode, flags, tlv_offset, seq_num, mep_id, md_name_format = struct.unpack_from(cls._PACK_STR, buf)
        md_name_length = 0
        md_name = b''
        md_lv = int(md_lv_version >> 5)
        version = int(md_lv_version & 31)
        rdi = int(flags >> 7)
        interval = int(flags & 7)
        offset = cls._MIN_LEN
        if md_name_format != cls._MD_FMT_NO_MD_NAME_PRESENT:
            md_name_length, = struct.unpack_from('!B', buf, offset)
            offset += cls._MD_NAME_LENGTH_LEN
            form = '%dB' % md_name_length
            md_name = struct.unpack_from(form, buf, offset)
            offset += md_name_length
        short_ma_name_format, short_ma_name_length = struct.unpack_from('!2B', buf, offset)
        offset += cls._SHORT_MA_NAME_FORMAT_LEN + cls._SHORT_MA_NAME_LENGTH_LEN
        form = '%dB' % short_ma_name_length
        short_ma_name = struct.unpack_from(form, buf, offset)
        offset = cls._MIN_LEN + (cls._MA_ID_LEN - cls._MD_NAME_FORMAT_LEN)
        tlvs = cls._parser_tlvs(buf[offset:])
        if md_name_format == cls._MD_FMT_DOMAIN_NAME_BASED_STRING or md_name_format == cls._MD_FMT_CHARACTER_STRING:
            md_name = b''.join(map(struct.Struct('>B').pack, md_name))
        if short_ma_name_format == cls._SHORT_MA_FMT_CHARACTER_STRING:
            short_ma_name = b''.join(map(struct.Struct('>B').pack, short_ma_name))
        return cls(md_lv, version, rdi, interval, seq_num, mep_id, md_name_format, md_name_length, md_name, short_ma_name_format, short_ma_name_length, short_ma_name, tlvs)

    def serialize(self):
        buf = struct.pack(self._PACK_STR, self.md_lv << 5 | self.version, self._opcode, self.rdi << 7 | self.interval, self._TLV_OFFSET, self.seq_num, self.mep_id, self.md_name_format)
        buf = bytearray(buf)
        if self.md_name_format != self._MD_FMT_NO_MD_NAME_PRESENT:
            if self.md_name_length == 0:
                self.md_name_length = len(self.md_name)
            buf.extend(struct.pack('!B%ds' % self.md_name_length, self.md_name_length, self.md_name))
        if self.short_ma_name_length == 0:
            self.short_ma_name_length = len(self.short_ma_name)
        buf.extend(struct.pack('!2B%ds' % self.short_ma_name_length, self.short_ma_name_format, self.short_ma_name_length, self.short_ma_name))
        maid_length = self._MD_NAME_FORMAT_LEN + self._SHORT_MA_NAME_FORMAT_LEN + self._SHORT_MA_NAME_LENGTH_LEN + self.short_ma_name_length
        if self.md_name_format != self._MD_FMT_NO_MD_NAME_PRESENT:
            maid_length += self._MD_NAME_LENGTH_LEN + self.md_name_length
        buf.extend(bytearray(self._MA_ID_LEN - maid_length))
        if self.tlvs:
            buf.extend(self._serialize_tlvs(self.tlvs))
        buf.extend(struct.pack('!B', CFM_END_TLV))
        return buf

    def __len__(self):
        return self._calc_len(self._MIN_LEN - self._MD_NAME_FORMAT_LEN + self._MA_ID_LEN)