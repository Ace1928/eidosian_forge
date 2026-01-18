from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
class ARC4Cipher:
    """ARC4 cipher object. Do not create it directly. Use
    :func:`Cryptodome.Cipher.ARC4.new` instead.
    """

    def __init__(self, key, *args, **kwargs):
        """Initialize an ARC4 cipher object

        See also `new()` at the module level."""
        if len(args) > 0:
            ndrop = args[0]
            args = args[1:]
        else:
            ndrop = kwargs.pop('drop', 0)
        if len(key) not in key_size:
            raise ValueError('Incorrect ARC4 key length (%d bytes)' % len(key))
        self._state = VoidPointer()
        result = _raw_arc4_lib.ARC4_stream_init(c_uint8_ptr(key), c_size_t(len(key)), self._state.address_of())
        if result != 0:
            raise ValueError('Error %d while creating the ARC4 cipher' % result)
        self._state = SmartPointer(self._state.get(), _raw_arc4_lib.ARC4_stream_destroy)
        if ndrop > 0:
            self.encrypt(b'\x00' * ndrop)
        self.block_size = 1
        self.key_size = len(key)

    def encrypt(self, plaintext):
        """Encrypt a piece of data.

        :param plaintext: The data to encrypt, of any size.
        :type plaintext: bytes, bytearray, memoryview
        :returns: the encrypted byte string, of equal length as the
          plaintext.
        """
        ciphertext = create_string_buffer(len(plaintext))
        result = _raw_arc4_lib.ARC4_stream_encrypt(self._state.get(), c_uint8_ptr(plaintext), ciphertext, c_size_t(len(plaintext)))
        if result:
            raise ValueError('Error %d while encrypting with RC4' % result)
        return get_raw_buffer(ciphertext)

    def decrypt(self, ciphertext):
        """Decrypt a piece of data.

        :param ciphertext: The data to decrypt, of any size.
        :type ciphertext: bytes, bytearray, memoryview
        :returns: the decrypted byte string, of equal length as the
          ciphertext.
        """
        try:
            return self.encrypt(ciphertext)
        except ValueError as e:
            raise ValueError(str(e).replace('enc', 'dec'))