import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
CFM (IEEE Std 802.1ag-2007) Reply Egress TLV encoder/decoder class.

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
    action            Egress Action.The default is 1 (EgrOK)
    mac_address       Egress MAC Address.
    port_id_length    Egress PortID Length.
                      (0 means automatically-calculate when encoding.)
    port_id_subtype   Egress PortID Subtype.
    port_id           Egress PortID.
    ================= =======================================
    