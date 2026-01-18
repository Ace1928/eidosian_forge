import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerSequence(DerObject):
    """Class to model a DER SEQUENCE.

        This object behaves like a dynamic Python sequence.

        Sub-elements that are INTEGERs behave like Python integers.

        Any other sub-element is a binary string encoded as a complete DER
        sub-element (TLV).

        An example of encoding is:

          >>> from Cryptodome.Util.asn1 import DerSequence, DerInteger
          >>> from binascii import hexlify, unhexlify
          >>> obj_der = unhexlify('070102')
          >>> seq_der = DerSequence([4])
          >>> seq_der.append(9)
          >>> seq_der.append(obj_der.encode())
          >>> print hexlify(seq_der.encode())

        which will show ``3009020104020109070102``, the DER encoding of the
        sequence containing ``4``, ``9``, and the object with payload ``02``.

        For decoding:

          >>> s = unhexlify(b'3009020104020109070102')
          >>> try:
          >>>   seq_der = DerSequence()
          >>>   seq_der.decode(s)
          >>>   print len(seq_der)
          >>>   print seq_der[0]
          >>>   print seq_der[:]
          >>> except ValueError:
          >>>   print "Not a valid DER SEQUENCE"

        the output will be::

          3
          4
          [4, 9, b'\x07\x01\x02']

        """

    def __init__(self, startSeq=None, implicit=None, explicit=None):
        """Initialize the DER object as a SEQUENCE.

                :Parameters:
                  startSeq : Python sequence
                    A sequence whose element are either integers or
                    other DER objects.

                  implicit : integer or byte
                    The IMPLICIT tag number (< 0x1F) to use for the encoded object.
                    It overrides the universal tag for SEQUENCE (16).
                    It cannot be combined with the ``explicit`` parameter.
                    By default, there is no IMPLICIT tag.

                  explicit : integer or byte
                    The EXPLICIT tag number (< 0x1F) to use for the encoded object.
                    It cannot be combined with the ``implicit`` parameter.
                    By default, there is no EXPLICIT tag.
                """
        DerObject.__init__(self, 16, b'', implicit, True, explicit)
        if startSeq is None:
            self._seq = []
        else:
            self._seq = startSeq

    def __delitem__(self, n):
        del self._seq[n]

    def __getitem__(self, n):
        return self._seq[n]

    def __setitem__(self, key, value):
        self._seq[key] = value

    def __setslice__(self, i, j, sequence):
        self._seq[i:j] = sequence

    def __delslice__(self, i, j):
        del self._seq[i:j]

    def __getslice__(self, i, j):
        return self._seq[max(0, i):max(0, j)]

    def __len__(self):
        return len(self._seq)

    def __iadd__(self, item):
        self._seq.append(item)
        return self

    def append(self, item):
        self._seq.append(item)
        return self

    def insert(self, index, item):
        self._seq.insert(index, item)
        return self

    def hasInts(self, only_non_negative=True):
        """Return the number of items in this sequence that are
                integers.

                Args:
                  only_non_negative (boolean):
                    If ``True``, negative integers are not counted in.
                """
        items = [x for x in self._seq if _is_number(x, only_non_negative)]
        return len(items)

    def hasOnlyInts(self, only_non_negative=True):
        """Return ``True`` if all items in this sequence are integers
                or non-negative integers.

                This function returns False is the sequence is empty,
                or at least one member is not an integer.

                Args:
                  only_non_negative (boolean):
                    If ``True``, the presence of negative integers
                    causes the method to return ``False``."""
        return self._seq and self.hasInts(only_non_negative) == len(self._seq)

    def encode(self):
        """Return this DER SEQUENCE, fully encoded as a
                binary string.

                Raises:
                  ValueError: if some elements in the sequence are neither integers
                              nor byte strings.
                """
        self.payload = b''
        for item in self._seq:
            if byte_string(item):
                self.payload += item
            elif _is_number(item):
                self.payload += DerInteger(item).encode()
            else:
                self.payload += item.encode()
        return DerObject.encode(self)

    def decode(self, der_encoded, strict=False, nr_elements=None, only_ints_expected=False):
        """Decode a complete DER SEQUENCE, and re-initializes this
                object with it.

                Args:
                  der_encoded (byte string):
                    A complete SEQUENCE DER element.
                  nr_elements (None or integer or list of integers):
                    The number of members the SEQUENCE can have
                  only_ints_expected (boolean):
                    Whether the SEQUENCE is expected to contain only integers.
                  strict (boolean):
                    Whether decoding must check for strict DER compliancy.

                Raises:
                  ValueError: in case of parsing errors.

                DER INTEGERs are decoded into Python integers. Any other DER
                element is not decoded. Its validity is not checked.
                """
        self._nr_elements = nr_elements
        result = DerObject.decode(self, der_encoded, strict=strict)
        if only_ints_expected and (not self.hasOnlyInts()):
            raise ValueError('Some members are not INTEGERs')
        return result

    def _decodeFromStream(self, s, strict):
        """Decode a complete DER SEQUENCE from a file."""
        self._seq = []
        DerObject._decodeFromStream(self, s, strict)
        p = BytesIO_EOF(self.payload)
        while p.remaining_data() > 0:
            p.set_bookmark()
            der = DerObject()
            der._decodeFromStream(p, strict)
            if der._tag_octet != 2:
                self._seq.append(p.data_since_bookmark())
            else:
                derInt = DerInteger()
                data = p.data_since_bookmark()
                derInt.decode(data, strict=strict)
                self._seq.append(derInt.value)
        ok = True
        if self._nr_elements is not None:
            try:
                ok = len(self._seq) in self._nr_elements
            except TypeError:
                ok = len(self._seq) == self._nr_elements
        if not ok:
            raise ValueError('Unexpected number of members (%d) in the sequence' % len(self._seq))