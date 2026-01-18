import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_heartbeat.register_param_type
@chunk_heartbeat_ack.register_param_type
class param_heartbeat(param):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Heartbeat Info Parameter (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.chunk_heartbeat
    - os_ken.lib.packet.sctp.chunk_heartbeat_ack

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    value          the sender-specific heartbeat information.
    length         length of this param containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """

    @classmethod
    def param_type(cls):
        return PTYPE_HEARTBEAT