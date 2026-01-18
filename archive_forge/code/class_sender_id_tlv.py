import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_SENDER_ID_TLV)
class sender_id_tlv(tlv):
    """CFM (IEEE Std 802.1ag-2007) Sender ID TLV encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ==================== =======================================
    Attribute            Description
    ==================== =======================================
    length               Length of Value field.
                         (0 means automatically-calculate when encoding.)
    chassis_id_length    Chassis ID Length.
                         (0 means automatically-calculate when encoding.)
    chassis_id_subtype   Chassis ID Subtype.
                         The default is 4 (Mac Address)
    chassis_id           Chassis ID.
    ma_domain_length     Management Address Domain Length.
                         (0 means automatically-calculate when encoding.)
    ma_domain            Management Address Domain.
    ma_length            Management Address Length.
                         (0 means automatically-calculate when encoding.)
    ma                   Management Address.
    ==================== =======================================
    """
    _PACK_STR = '!BHB'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _CHASSIS_ID_LENGTH_LEN = 1
    _CHASSIS_ID_SUBTYPE_LEN = 1
    _MA_DOMAIN_LENGTH_LEN = 1
    _MA_LENGTH_LEN = 1
    _CHASSIS_ID_CHASSIS_COMPONENT = 1
    _CHASSIS_ID_INTERFACE_ALIAS = 2
    _CHASSIS_ID_PORT_COMPONENT = 3
    _CHASSIS_ID_MAC_ADDRESS = 4
    _CHASSIS_ID_NETWORK_ADDRESS = 5
    _CHASSIS_ID_INTERFACE_NAME = 6
    _CHASSIS_ID_LOCALLY_ASSIGNED = 7

    def __init__(self, length=0, chassis_id_length=0, chassis_id_subtype=_CHASSIS_ID_MAC_ADDRESS, chassis_id=b'', ma_domain_length=0, ma_domain=b'', ma_length=0, ma=b''):
        super(sender_id_tlv, self).__init__(length)
        self._type = CFM_SENDER_ID_TLV
        self.chassis_id_length = chassis_id_length
        assert chassis_id_subtype in [self._CHASSIS_ID_CHASSIS_COMPONENT, self._CHASSIS_ID_INTERFACE_ALIAS, self._CHASSIS_ID_PORT_COMPONENT, self._CHASSIS_ID_MAC_ADDRESS, self._CHASSIS_ID_NETWORK_ADDRESS, self._CHASSIS_ID_INTERFACE_NAME, self._CHASSIS_ID_LOCALLY_ASSIGNED]
        self.chassis_id_subtype = chassis_id_subtype
        self.chassis_id = chassis_id
        self.ma_domain_length = ma_domain_length
        self.ma_domain = ma_domain
        self.ma_length = ma_length
        self.ma = ma

    @classmethod
    def parser(cls, buf):
        type_, length, chassis_id_length = struct.unpack_from(cls._PACK_STR, buf)
        chassis_id_subtype = 4
        chassis_id = b''
        ma_domain_length = 0
        ma_domain = b''
        ma_length = 0
        ma = b''
        offset = cls._MIN_LEN
        if chassis_id_length != 0:
            chassis_id_subtype, = struct.unpack_from('!B', buf, offset)
            offset += cls._CHASSIS_ID_SUBTYPE_LEN
            form = '%ds' % chassis_id_length
            chassis_id, = struct.unpack_from(form, buf, offset)
            offset += chassis_id_length
        if length + (cls._TYPE_LEN + cls._LENGTH_LEN) > offset:
            ma_domain_length, = struct.unpack_from('!B', buf, offset)
            offset += cls._MA_DOMAIN_LENGTH_LEN
            form = '%ds' % ma_domain_length
            ma_domain, = struct.unpack_from(form, buf, offset)
            offset += ma_domain_length
            if length + (cls._TYPE_LEN + cls._LENGTH_LEN) > offset:
                ma_length, = struct.unpack_from('!B', buf, offset)
                offset += cls._MA_LENGTH_LEN
                form = '%ds' % ma_length
                ma, = struct.unpack_from(form, buf, offset)
        return cls(length, chassis_id_length, chassis_id_subtype, chassis_id, ma_domain_length, ma_domain, ma_length, ma)

    def serialize(self):
        if self.chassis_id_length == 0:
            self.chassis_id_length = len(self.chassis_id)
        if self.ma_domain_length == 0:
            self.ma_domain_length = len(self.ma_domain)
        if self.ma_length == 0:
            self.ma_length = len(self.ma)
        if self.length == 0:
            self.length += self._CHASSIS_ID_LENGTH_LEN
            if self.chassis_id_length != 0:
                self.length += self._CHASSIS_ID_SUBTYPE_LEN + self.chassis_id_length
            if self.chassis_id_length != 0 or self.ma_domain_length != 0:
                self.length += self._MA_DOMAIN_LENGTH_LEN
            if self.ma_domain_length != 0:
                self.length += self.ma_domain_length + self._MA_LENGTH_LEN + self.ma_length
        buf = struct.pack(self._PACK_STR, self._type, self.length, self.chassis_id_length)
        buf = bytearray(buf)
        if self.chassis_id_length != 0:
            buf.extend(struct.pack('!B', self.chassis_id_subtype))
            form = '%ds' % self.chassis_id_length
            buf.extend(struct.pack(form, self.chassis_id))
        if self.chassis_id_length != 0 or self.ma_domain_length != 0:
            buf.extend(struct.pack('!B', self.ma_domain_length))
        if self.ma_domain_length != 0:
            form = '%ds' % self.ma_domain_length
            buf.extend(struct.pack(form, self.ma_domain))
            buf.extend(struct.pack('!B', self.ma_length))
            if self.ma_length != 0:
                form = '%ds' % self.ma_length
                buf.extend(struct.pack(form, self.ma))
        return buf