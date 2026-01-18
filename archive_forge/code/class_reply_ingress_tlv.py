import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_REPLY_INGRESS_TLV)
class reply_ingress_tlv(reply_tlv):
    """CFM (IEEE Std 802.1ag-2007) Reply Ingress TLV encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ================= =======================================
    Attribute         Description
    ================= =======================================
    length            Length of Value field.
                      (0 means automatically-calculate when encoding.)
    action            Ingress Action.The default is 1 (IngOK)
    mac_address       Ingress MAC Address.
    port_id_length    Ingress PortID Length.
                      (0 means automatically-calculate when encoding.)
    port_id_subtype   Ingress PortID Subtype.
    port_id           Ingress PortID.
    ================= =======================================
    """
    _ING_OK = 1
    _ING_DOWN = 2
    _ING_BLOCKED = 3
    _ING_VID = 4

    def __init__(self, length=0, action=_ING_OK, mac_address='00:00:00:00:00:00', port_id_length=0, port_id_subtype=0, port_id=b''):
        super(reply_ingress_tlv, self).__init__(length, action, mac_address, port_id_length, port_id_subtype, port_id)
        assert action in [self._ING_OK, self._ING_DOWN, self._ING_BLOCKED, self._ING_VID]
        self._type = CFM_REPLY_INGRESS_TLV