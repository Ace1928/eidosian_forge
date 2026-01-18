import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerBoolean(DerObject):
    """Class to model a DER-encoded BOOLEAN.

    An example of encoding is::

    >>> from Cryptodome.Util.asn1 import DerBoolean
    >>> bool_der = DerBoolean(True)
    >>> print(bool_der.encode().hex())

    which will show ``0101ff``, the DER encoding of True.

    And for decoding::

    >>> s = bytes.fromhex('0101ff')
    >>> try:
    >>>   bool_der = DerBoolean()
    >>>   bool_der.decode(s)
    >>>   print(bool_der.value)
    >>> except ValueError:
    >>>   print "Not a valid DER BOOLEAN"

    the output will be ``True``.

    :ivar value: The boolean value
    :vartype value: boolean
    """

    def __init__(self, value=False, implicit=None, explicit=None):
        """Initialize the DER object as a BOOLEAN.

        Args:
          value (boolean):
            The value of the boolean. Default is False.

          implicit (integer or byte):
            The IMPLICIT tag number (< 0x1F) to use for the encoded object.
            It overrides the universal tag for BOOLEAN (1).
            It cannot be combined with the ``explicit`` parameter.
            By default, there is no IMPLICIT tag.

          explicit (integer or byte):
            The EXPLICIT tag number (< 0x1F) to use for the encoded object.
            It cannot be combined with the ``implicit`` parameter.
            By default, there is no EXPLICIT tag.
        """
        DerObject.__init__(self, 1, b'', implicit, False, explicit)
        self.value = value

    def encode(self):
        """Return the DER BOOLEAN, fully encoded as a binary string."""
        self.payload = b'\xff' if self.value else b'\x00'
        return DerObject.encode(self)

    def decode(self, der_encoded, strict=False):
        """Decode a DER-encoded BOOLEAN, and re-initializes this object with it.

        Args:
            der_encoded (byte string): A DER-encoded BOOLEAN.

        Raises:
            ValueError: in case of parsing errors.
        """
        return DerObject.decode(self, der_encoded, strict)

    def _decodeFromStream(self, s, strict):
        """Decode a DER-encoded BOOLEAN from a file."""
        DerObject._decodeFromStream(self, s, strict)
        if len(self.payload) != 1:
            raise ValueError('Invalid encoding for DER BOOLEAN: payload is not 1 byte')
        if bord(self.payload[0]) == 0:
            self.value = False
        elif bord(self.payload[0]) == 255:
            self.value = True
        else:
            raise ValueError('Invalid payload for DER BOOLEAN')