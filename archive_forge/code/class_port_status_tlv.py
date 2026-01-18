import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_PORT_STATUS_TLV)
class port_status_tlv(tlv):
    """CFM (IEEE Std 802.1ag-2007) Port Status TLV encoder/decoder class.

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
    port_status          Port Status.The default is 1 (psUp)
    ==================== =======================================
    """
    _PACK_STR = '!BHB'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _PS_BLOCKED = 1
    _PS_UP = 2

    def __init__(self, length=0, port_status=_PS_UP):
        super(port_status_tlv, self).__init__(length)
        self._type = CFM_PORT_STATUS_TLV
        assert port_status in [self._PS_BLOCKED, self._PS_UP]
        self.port_status = port_status

    @classmethod
    def parser(cls, buf):
        type_, length, port_status = struct.unpack_from(cls._PACK_STR, buf)
        return cls(length, port_status)

    def serialize(self):
        if self.length == 0:
            self.length = 1
        buf = struct.pack(self._PACK_STR, self._type, self.length, self.port_status)
        return bytearray(buf)