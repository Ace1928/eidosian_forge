from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash.keccak import _raw_keccak_lib
class SHAKE128_XOF(object):
    """A SHAKE128 hash object.
    Do not instantiate directly.
    Use the :func:`new` function.

    :ivar oid: ASN.1 Object ID
    :vartype oid: string
    """
    oid = '2.16.840.1.101.3.4.2.11'

    def __init__(self, data=None):
        state = VoidPointer()
        result = _raw_keccak_lib.keccak_init(state.address_of(), c_size_t(32), c_ubyte(24))
        if result:
            raise ValueError('Error %d while instantiating SHAKE128' % result)
        self._state = SmartPointer(state.get(), _raw_keccak_lib.keccak_destroy)
        self._is_squeezing = False
        self._padding = 31
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """
        if self._is_squeezing:
            raise TypeError("You cannot call 'update' after the first 'read'")
        result = _raw_keccak_lib.keccak_absorb(self._state.get(), c_uint8_ptr(data), c_size_t(len(data)))
        if result:
            raise ValueError('Error %d while updating SHAKE128 state' % result)
        return self

    def read(self, length):
        """
        Compute the next piece of XOF output.

        .. note::
            You cannot use :meth:`update` anymore after the first call to
            :meth:`read`.

        Args:
            length (integer): the amount of bytes this method must return

        :return: the next piece of XOF output (of the given length)
        :rtype: byte string
        """
        self._is_squeezing = True
        bfr = create_string_buffer(length)
        result = _raw_keccak_lib.keccak_squeeze(self._state.get(), bfr, c_size_t(length), c_ubyte(self._padding))
        if result:
            raise ValueError('Error %d while extracting from SHAKE128' % result)
        return get_raw_buffer(bfr)

    def new(self, data=None):
        return type(self)(data=data)