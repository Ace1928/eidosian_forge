import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
@bfd.register_auth_type(BFD_AUTH_METICULOUS_KEYED_MD5)
class MeticulousKeyedMD5(KeyedMD5):
    """ BFD (RFC 5880) Meticulous Keyed MD5 Authentication Section class

    All methods of this class are inherited from ``KeyedMD5``.

    An instance has the following attributes.
    Most of them are same to the on-wire counterparts but in host byte order.

    .. tabularcolumns:: |l|L|

    =========== =================================================
    Attribute   Description
    =========== =================================================
    auth_type   (Fixed) The authentication type in use.
    auth_key_id The authentication Key ID in use.
    seq         The sequence number for this packet.
                This value is incremented for each
                successive packet transmitted for a session.
    auth_key    The shared MD5 key for this packet.
    digest      (Optional) The 16-byte MD5 digest for the packet.
    auth_len    (Fixed) The length of the authentication section
                is 24 bytes.
    =========== =================================================
    """
    pass