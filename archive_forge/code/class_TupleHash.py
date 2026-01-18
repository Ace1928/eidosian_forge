from Cryptodome.Util.py3compat import bord, is_bytes, tobytes
from . import cSHAKE128
from .cSHAKE128 import _encode_str, _right_encode
class TupleHash(object):
    """A Tuple hash object.
    Do not instantiate directly.
    Use the :func:`new` function.
    """

    def __init__(self, custom, cshake, digest_size):
        self.digest_size = digest_size
        self._cshake = cshake._new(b'', custom, b'TupleHash')
        self._digest = None

    def update(self, *data):
        """Authenticate the next tuple of byte strings.
        TupleHash guarantees the logical separation between each byte string.

        Args:
            data (bytes/bytearray/memoryview): One or more items to hash.
        """
        if self._digest is not None:
            raise TypeError("You cannot call 'update' after 'digest' or 'hexdigest'")
        for item in data:
            if not is_bytes(item):
                raise TypeError("You can only call 'update' on bytes")
            self._cshake.update(_encode_str(item))
        return self

    def digest(self):
        """Return the **binary** (non-printable) digest of the tuple of byte strings.

        :return: The hash digest. Binary form.
        :rtype: byte string
        """
        if self._digest is None:
            self._cshake.update(_right_encode(self.digest_size * 8))
            self._digest = self._cshake.read(self.digest_size)
        return self._digest

    def hexdigest(self):
        """Return the **printable** digest of the tuple of byte strings.

        :return: The hash digest. Hexadecimal encoded.
        :rtype: string
        """
        return ''.join(['%02x' % bord(x) for x in tuple(self.digest())])

    def new(self, **kwargs):
        """Return a new instance of a TupleHash object.
        See :func:`new`.
        """
        if 'digest_bytes' not in kwargs and 'digest_bits' not in kwargs:
            kwargs['digest_bytes'] = self.digest_size
        return new(**kwargs)