import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
@bfd.register_auth_type(BFD_AUTH_KEYED_MD5)
class KeyedMD5(BFDAuth):
    """ BFD (RFC 5880) Keyed MD5 Authentication Section class

    An instance has the following attributes.
    Most of them are same to the on-wire counterparts but in host byte order.

    .. tabularcolumns:: |l|L|

    =========== =================================================
    Attribute   Description
    =========== =================================================
    auth_type   (Fixed) The authentication type in use.
    auth_key_id The authentication Key ID in use.
    seq         The sequence number for this packet.
                This value is incremented occasionally.
    auth_key    The shared MD5 key for this packet.
    digest      (Optional) The 16-byte MD5 digest for the packet.
    auth_len    (Fixed) The length of the authentication section
                is 24 bytes.
    =========== =================================================
    """
    _PACK_STR = '!BBL16s'
    _PACK_STR_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, auth_key_id, seq, auth_key=None, digest=None, auth_len=None):
        self.auth_key_id = auth_key_id
        self.seq = seq
        self.auth_key = auth_key
        self.digest = digest
        super(KeyedMD5, self).__init__(auth_len)

    def __len__(self):
        return 24

    @classmethod
    def parser(cls, buf):
        auth_type, auth_len = cls.parser_hdr(buf)
        assert auth_type == cls.auth_type
        assert auth_len == 24
        auth_key_id, reserved, seq, digest = struct.unpack_from(cls._PACK_STR, buf[cls._PACK_HDR_STR_LEN:])
        assert reserved == 0
        msg = cls(auth_key_id=auth_key_id, seq=seq, auth_key=None, digest=digest)
        return (msg, None, None)

    def serialize(self, payload, prev):
        """Encode a Keyed MD5 Authentication Section.

        This method is used only when encoding an BFD Control packet.

        ``payload`` is the rest of the packet which will immediately follow
        this section.

        ``prev`` is a ``bfd`` instance for the BFD Control header which this
        authentication section belongs to. It's necessary to be assigned
        because an MD5 digest must be calculated over the entire BFD Control
        packet.
        """
        assert self.auth_key is not None and len(self.auth_key) <= 16
        assert isinstance(prev, bfd)
        bfd_bin = prev.pack()
        auth_hdr_bin = self.serialize_hdr()
        auth_data_bin = struct.pack(self._PACK_STR, self.auth_key_id, 0, self.seq, self.auth_key + b'\x00' * (len(self.auth_key) - 16))
        h = hashlib.md5()
        h.update(bfd_bin + auth_hdr_bin + auth_data_bin)
        self.digest = h.digest()
        return auth_hdr_bin + struct.pack(self._PACK_STR, self.auth_key_id, 0, self.seq, self.digest)

    def authenticate(self, prev, auth_keys=None):
        """Authenticate the MD5 digest for this packet.

        This method can be invoked only when ``self.digest`` is defined.

        Returns a boolean indicates whether the digest can be authenticated
        by the correspondent Auth Key or not.

        ``prev`` is a ``bfd`` instance for the BFD Control header which this
        authentication section belongs to. It's necessary to be assigned
        because an MD5 digest must be calculated over the entire BFD Control
        packet.

        ``auth_keys`` is a dictionary of authentication key chain which
        key is an integer of *Auth Key ID* and value is a string of *Auth Key*.
        """
        auth_keys = auth_keys if auth_keys else {}
        assert isinstance(prev, bfd)
        if self.digest is None:
            return False
        if self.auth_key_id not in auth_keys:
            return False
        auth_key = auth_keys[self.auth_key_id]
        bfd_bin = prev.pack()
        auth_hdr_bin = self.serialize_hdr()
        auth_data_bin = struct.pack(self._PACK_STR, self.auth_key_id, 0, self.seq, auth_key + b'\x00' * (len(auth_key) - 16))
        h = hashlib.md5()
        h.update(bfd_bin + auth_hdr_bin + auth_data_bin)
        if self.digest == h.digest():
            return True
        else:
            return False