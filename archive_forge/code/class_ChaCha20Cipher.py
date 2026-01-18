from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
class ChaCha20Cipher(object):
    """ChaCha20 (or XChaCha20) cipher object.
    Do not create it directly. Use :py:func:`new` instead.

    :var nonce: The nonce with length 8, 12 or 24 bytes
    :vartype nonce: bytes
    """
    block_size = 1

    def __init__(self, key, nonce):
        """Initialize a ChaCha20/XChaCha20 cipher object

        See also `new()` at the module level."""
        self.nonce = _copy_bytes(None, None, nonce)
        if len(nonce) == 24:
            key = _HChaCha20(key, nonce[:16])
            nonce = b'\x00' * 4 + nonce[16:]
            self._name = 'XChaCha20'
        else:
            self._name = 'ChaCha20'
            nonce = self.nonce
        self._next = ('encrypt', 'decrypt')
        self._state = VoidPointer()
        result = _raw_chacha20_lib.chacha20_init(self._state.address_of(), c_uint8_ptr(key), c_size_t(len(key)), nonce, c_size_t(len(nonce)))
        if result:
            raise ValueError('Error %d instantiating a %s cipher' % (result, self._name))
        self._state = SmartPointer(self._state.get(), _raw_chacha20_lib.chacha20_destroy)

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
        if 'encrypt' not in self._next:
            raise TypeError('Cipher object can only be used for decryption')
        self._next = ('encrypt',)
        return self._encrypt(plaintext, output)

    def _encrypt(self, plaintext, output):
        """Encrypt without FSM checks"""
        if output is None:
            ciphertext = create_string_buffer(len(plaintext))
        else:
            ciphertext = output
            if not is_writeable_buffer(output):
                raise TypeError('output must be a bytearray or a writeable memoryview')
            if len(plaintext) != len(output):
                raise ValueError('output must have the same length as the input  (%d bytes)' % len(plaintext))
        result = _raw_chacha20_lib.chacha20_encrypt(self._state.get(), c_uint8_ptr(plaintext), c_uint8_ptr(ciphertext), c_size_t(len(plaintext)))
        if result:
            raise ValueError('Error %d while encrypting with %s' % (result, self._name))
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
        if 'decrypt' not in self._next:
            raise TypeError('Cipher object can only be used for encryption')
        self._next = ('decrypt',)
        try:
            return self._encrypt(ciphertext, output)
        except ValueError as e:
            raise ValueError(str(e).replace('enc', 'dec'))

    def seek(self, position):
        """Seek to a certain position in the key stream.

        Args:
          position (integer):
            The absolute position within the key stream, in bytes.
        """
        position, offset = divmod(position, 64)
        block_low = position & 4294967295
        block_high = position >> 32
        result = _raw_chacha20_lib.chacha20_seek(self._state.get(), c_ulong(block_high), c_ulong(block_low), offset)
        if result:
            raise ValueError('Error %d while seeking with %s' % (result, self._name))