import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
class BFDAuth(stringify.StringifyMixin):
    """Base class of BFD (RFC 5880) Authentication Section

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.

    .. tabularcolumns:: |l|L|

    =========== ============================================
    Attribute   Description
    =========== ============================================
    auth_type   The authentication type in use.
    auth_len    The length, in bytes, of the authentication
                section, including the ``auth_type`` and
                ``auth_len`` fields.
    =========== ============================================
    """
    _PACK_HDR_STR = '!BB'
    _PACK_HDR_STR_LEN = struct.calcsize(_PACK_HDR_STR)
    auth_type = None

    def __init__(self, auth_len=None):
        super(BFDAuth, self).__init__()
        if isinstance(auth_len, int):
            self.auth_len = auth_len
        else:
            self.auth_len = len(self)

    @staticmethod
    def set_type(subcls, auth_type):
        assert issubclass(subcls, BFDAuth)
        subcls.auth_type = auth_type

    @classmethod
    def parser_hdr(cls, buf):
        """
        Parser for common part of authentication section.
        """
        return struct.unpack_from(cls._PACK_HDR_STR, buf[:cls._PACK_HDR_STR_LEN])

    def serialize_hdr(self):
        """
        Serialization function for common part of authentication section.
        """
        return struct.pack(self._PACK_HDR_STR, self.auth_type, self.auth_len)