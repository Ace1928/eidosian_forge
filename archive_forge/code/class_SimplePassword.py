import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
@bfd.register_auth_type(BFD_AUTH_SIMPLE_PASS)
class SimplePassword(BFDAuth):
    """ BFD (RFC 5880) Simple Password Authentication Section class

    An instance has the following attributes.
    Most of them are same to the on-wire counterparts but in host byte order.

    .. tabularcolumns:: |l|L|

    =========== ============================================
    Attribute   Description
    =========== ============================================
    auth_type   (Fixed) The authentication type in use.
    auth_key_id The authentication Key ID in use.
    password    The simple password in use on this session.
                The password is a binary string, and MUST be
                from 1 to 16 bytes in length.
    auth_len    The length, in bytes, of the authentication
                section, including the ``auth_type`` and
                ``auth_len`` fields.
    =========== ============================================
    """
    _PACK_STR = '!B'
    _PACK_STR_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, auth_key_id, password, auth_len=None):
        assert len(password) >= 1 and len(password) <= 16
        self.auth_key_id = auth_key_id
        self.password = password
        super(SimplePassword, self).__init__(auth_len)

    def __len__(self):
        return self._PACK_HDR_STR_LEN + self._PACK_STR_LEN + len(self.password)

    @classmethod
    def parser(cls, buf):
        auth_type, auth_len = cls.parser_hdr(buf)
        assert auth_type == cls.auth_type
        auth_key_id = operator.getitem(buf, cls._PACK_HDR_STR_LEN)
        password = buf[cls._PACK_HDR_STR_LEN + cls._PACK_STR_LEN:auth_len]
        msg = cls(auth_key_id, password, auth_len)
        return (msg, None, None)

    def serialize(self, payload, prev):
        """Encode a Simple Password Authentication Section.

        ``payload`` is the rest of the packet which will immediately follow
        this section.

        ``prev`` is a ``bfd`` instance for the BFD Control header. It's not
        necessary for encoding only the Simple Password section.
        """
        return self.serialize_hdr() + struct.pack(self._PACK_STR, self.auth_key_id) + self.password

    def authenticate(self, prev=None, auth_keys=None):
        """Authenticate the password for this packet.

        This method can be invoked only when ``self.password`` is defined.

        Returns a boolean indicates whether the password can be authenticated
        or not.

        ``prev`` is a ``bfd`` instance for the BFD Control header. It's not
        necessary for authenticating the Simple Password.

        ``auth_keys`` is a dictionary of authentication key chain which
        key is an integer of *Auth Key ID* and value is a string of *Password*.
        """
        auth_keys = auth_keys if auth_keys else {}
        assert isinstance(prev, bfd)
        if self.auth_key_id in auth_keys and self.password == auth_keys[self.auth_key_id]:
            return True
        else:
            return False