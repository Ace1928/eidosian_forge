from Cryptodome.Util.py3compat import bchr, concat_buffers
from Cryptodome.Util._raw_api import (VoidPointer, SmartPointer,
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Hash.keccak import _raw_keccak_lib
class cSHAKE_XOF(object):
    """A cSHAKE hash object.
    Do not instantiate directly.
    Use the :func:`new` function.
    """

    def __init__(self, data, custom, capacity, function):
        state = VoidPointer()
        if custom or function:
            prefix_unpad = _encode_str(function) + _encode_str(custom)
            prefix = _bytepad(prefix_unpad, (1600 - capacity) // 8)
            self._padding = 4
        else:
            prefix = None
            self._padding = 31
        result = _raw_keccak_lib.keccak_init(state.address_of(), c_size_t(capacity // 8), c_ubyte(24))
        if result:
            raise ValueError('Error %d while instantiating cSHAKE' % result)
        self._state = SmartPointer(state.get(), _raw_keccak_lib.keccak_destroy)
        self._is_squeezing = False
        if prefix:
            self.update(prefix)
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
            raise ValueError('Error %d while updating %s state' % (result, self.name))
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
            raise ValueError('Error %d while extracting from %s' % (result, self.name))
        return get_raw_buffer(bfr)