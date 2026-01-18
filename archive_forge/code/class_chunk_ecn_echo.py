import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_ecn_echo(chunk_ecn_base):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for ECN-Echo chunk (RFC 4960 Appendix A.).

    This class is used with the following.

    - os_ken.lib.packet.sctp.sctp

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    flags          set to '0'. this field will be ignored.
    length         length of this chunk containing this header.
                   (0 means automatically-calculate when encoding)
    low_tsn        the lowest TSN.
    ============== =====================================================
    """

    @classmethod
    def chunk_type(cls):
        return TYPE_ECN_ECHO