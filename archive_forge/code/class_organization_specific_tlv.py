import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_ORGANIZATION_SPECIFIC_TLV)
class organization_specific_tlv(tlv):
    """CFM (IEEE Std 802.1ag-2007) Organization Specific TLV
       encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =======================================
    Attribute      Description
    ============== =======================================
    length         Length of Value field.
                   (0 means automatically-calculate when encoding.)
    oui            Organizationally Unique Identifier.
    subtype        Subtype.
    value          Value.(optional)
    ============== =======================================
    """
    _PACK_STR = '!BH3sB'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _OUI_AND_SUBTYPE_LEN = 4

    def __init__(self, length=0, oui=b'\x00\x00\x00', subtype=0, value=b''):
        super(organization_specific_tlv, self).__init__(length)
        self._type = CFM_ORGANIZATION_SPECIFIC_TLV
        self.oui = oui
        self.subtype = subtype
        self.value = value

    @classmethod
    def parser(cls, buf):
        type_, length, oui, subtype = struct.unpack_from(cls._PACK_STR, buf)
        value = b''
        if length > cls._OUI_AND_SUBTYPE_LEN:
            form = '%ds' % (length - cls._OUI_AND_SUBTYPE_LEN)
            value, = struct.unpack_from(form, buf, cls._MIN_LEN)
        return cls(length, oui, subtype, value)

    def serialize(self):
        if self.length == 0:
            self.length = len(self.value) + self._OUI_AND_SUBTYPE_LEN
        buf = struct.pack(self._PACK_STR, self._type, self.length, self.oui, self.subtype)
        buf = bytearray(buf)
        form = '%ds' % (self.length - self._OUI_AND_SUBTYPE_LEN)
        buf.extend(struct.pack(form, self.value))
        return buf