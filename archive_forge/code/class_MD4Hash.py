from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
class MD4Hash(object):
    """Class that implements an MD4 hash
    """
    digest_size = 16
    block_size = 64
    oid = '1.2.840.113549.2.4'

    def __init__(self, data=None):
        state = VoidPointer()
        result = _raw_md4_lib.md4_init(state.address_of())
        if result:
            raise ValueError('Error %d while instantiating MD4' % result)
        self._state = SmartPointer(state.get(), _raw_md4_lib.md4_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Repeated calls are equivalent to a single call with the concatenation
        of all the arguments. In other words:

           >>> m.update(a); m.update(b)

        is equivalent to:

           >>> m.update(a+b)

        :Parameters:
          data : byte string/byte array/memoryview
            The next chunk of the message being hashed.
        """
        result = _raw_md4_lib.md4_update(self._state.get(), c_uint8_ptr(data), c_size_t(len(data)))
        if result:
            raise ValueError('Error %d while instantiating MD4' % result)

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that
        has been hashed so far.

        This method does not change the state of the hash object.
        You can continue updating the object after calling this function.

        :Return: A byte string of `digest_size` bytes. It may contain non-ASCII
         characters, including null bytes.
        """
        bfr = create_string_buffer(self.digest_size)
        result = _raw_md4_lib.md4_digest(self._state.get(), bfr)
        if result:
            raise ValueError('Error %d while instantiating MD4' % result)
        return get_raw_buffer(bfr)

    def hexdigest(self):
        """Return the **printable** digest of the message that has been
        hashed so far.

        This method does not change the state of the hash object.

        :Return: A string of 2* `digest_size` characters. It contains only
         hexadecimal ASCII digits.
        """
        return ''.join(['%02x' % bord(x) for x in self.digest()])

    def copy(self):
        """Return a copy ("clone") of the hash object.

        The copy will have the same internal state as the original hash
        object.
        This can be used to efficiently compute the digests of strings that
        share a common initial substring.

        :Return: A hash object of the same type
        """
        clone = MD4Hash()
        result = _raw_md4_lib.md4_copy(self._state.get(), clone._state.get())
        if result:
            raise ValueError('Error %d while copying MD4' % result)
        return clone

    def new(self, data=None):
        return MD4Hash(data)