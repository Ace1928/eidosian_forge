from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Random import get_random_bytes
class Salsa20Cipher:
    """Salsa20 cipher object. Do not create it directly. Use :py:func:`new`
    instead.

    :var nonce: The nonce with length 8
    :vartype nonce: byte string
    """

    def __init__(self, key, nonce):
        """Initialize a Salsa20 cipher object

        See also `new()` at the module level."""
        if len(key) not in key_size:
            raise ValueError('Incorrect key length for Salsa20 (%d bytes)' % len(key))
        if len(nonce) != 8:
            raise ValueError('Incorrect nonce length for Salsa20 (%d bytes)' % len(nonce))
        self.nonce = _copy_bytes(None, None, nonce)
        self._state = VoidPointer()
        result = _raw_salsa20_lib.Salsa20_stream_init(c_uint8_ptr(key), c_size_t(len(key)), c_uint8_ptr(nonce), c_size_t(len(nonce)), self._state.address_of())
        if result:
            raise ValueError('Error %d instantiating a Salsa20 cipher')
        self._state = SmartPointer(self._state.get(), _raw_salsa20_lib.Salsa20_stream_destroy)
        self.block_size = 1
        self.key_size = len(key)

    def encrypt(self, plaintext, output=None):
        """Encrypt a piece of data.

        Args:
          plaintext(bytes/bytearray/memoryview): The data to encrypt, of any size.
        Keyword Args:
          output(bytes/bytearray/memoryview): The location where the ciphertext
            is written to. If ``None``, the ciphertext is returned.
        Returns:
          If ``output`` is ``None``, the ciphertext is returned as ``bytes``.
          Otherwise, ``None``.
        """
        if output is None:
            ciphertext = create_string_buffer(len(plaintext))
        else:
            ciphertext = output
            if not is_writeable_buffer(output):
                raise TypeError('output must be a bytearray or a writeable memoryview')
            if len(plaintext) != len(output):
                raise ValueError('output must have the same length as the input  (%d bytes)' % len(plaintext))
        result = _raw_salsa20_lib.Salsa20_stream_encrypt(self._state.get(), c_uint8_ptr(plaintext), c_uint8_ptr(ciphertext), c_size_t(len(plaintext)))
        if result:
            raise ValueError('Error %d while encrypting with Salsa20' % result)
        if output is None:
            return get_raw_buffer(ciphertext)
        else:
            return None

    def decrypt(self, ciphertext, output=None):
        """Decrypt a piece of data.
        
        Args:
          ciphertext(bytes/bytearray/memoryview): The data to decrypt, of any size.
        Keyword Args:
          output(bytes/bytearray/memoryview): The location where the plaintext
            is written to. If ``None``, the plaintext is returned.
        Returns:
          If ``output`` is ``None``, the plaintext is returned as ``bytes``.
          Otherwise, ``None``.
        """
        try:
            return self.encrypt(ciphertext, output=output)
        except ValueError as e:
            raise ValueError(str(e).replace('enc', 'dec'))