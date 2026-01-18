from Cryptodome.Util.py3compat import _copy_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Random import get_random_bytes
class OfbMode(object):
    """*Output FeedBack (OFB)*.

    This mode is very similar to CBC, but it
    transforms the underlying block cipher into a stream cipher.

    The keystream is the iterated block encryption of the
    previous ciphertext block.

    An Initialization Vector (*IV*) is required.

    See `NIST SP800-38A`_ , Section 6.4.

    .. _`NIST SP800-38A` : http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf

    :undocumented: __init__
    """

    def __init__(self, block_cipher, iv):
        """Create a new block cipher, configured in OFB mode.

        :Parameters:
          block_cipher : C pointer
            A smart pointer to the low-level block cipher instance.

          iv : bytes/bytearray/memoryview
            The initialization vector to use for encryption or decryption.
            It is as long as the cipher block.

            **The IV must be a nonce, to to be reused for any other
            message**. It shall be a nonce or a random value.

            Reusing the *IV* for encryptions performed with the same key
            compromises confidentiality.
        """
        self._state = VoidPointer()
        result = raw_ofb_lib.OFB_start_operation(block_cipher.get(), c_uint8_ptr(iv), c_size_t(len(iv)), self._state.address_of())
        if result:
            raise ValueError('Error %d while instantiating the OFB mode' % result)
        self._state = SmartPointer(self._state.get(), raw_ofb_lib.OFB_stop_operation)
        block_cipher.release()
        self.block_size = len(iv)
        'The block size of the underlying cipher, in bytes.'
        self.iv = _copy_bytes(None, None, iv)
        'The Initialization Vector originally used to create the object.\n        The value does not change.'
        self.IV = self.iv
        'Alias for `iv`'
        self._next = ['encrypt', 'decrypt']

    def encrypt(self, plaintext, output=None):
        """Encrypt data with the key and the parameters set at initialization.

        A cipher object is stateful: once you have encrypted a message
        you cannot encrypt (or decrypt) another message using the same
        object.

        The data to encrypt can be broken up in two or
        more pieces and `encrypt` can be called multiple times.

        That is, the statement:

            >>> c.encrypt(a) + c.encrypt(b)

        is equivalent to:

             >>> c.encrypt(a+b)

        This function does not add any padding to the plaintext.

        :Parameters:
          plaintext : bytes/bytearray/memoryview
            The piece of data to encrypt.
            It can be of any length.
        :Keywords:
          output : bytearray/memoryview
            The location where the ciphertext must be written to.
            If ``None``, the ciphertext is returned.
        :Return:
          If ``output`` is ``None``, the ciphertext is returned as ``bytes``.
          Otherwise, ``None``.
        """
        if 'encrypt' not in self._next:
            raise TypeError('encrypt() cannot be called after decrypt()')
        self._next = ['encrypt']
        if output is None:
            ciphertext = create_string_buffer(len(plaintext))
        else:
            ciphertext = output
            if not is_writeable_buffer(output):
                raise TypeError('output must be a bytearray or a writeable memoryview')
            if len(plaintext) != len(output):
                raise ValueError('output must have the same length as the input  (%d bytes)' % len(plaintext))
        result = raw_ofb_lib.OFB_encrypt(self._state.get(), c_uint8_ptr(plaintext), c_uint8_ptr(ciphertext), c_size_t(len(plaintext)))
        if result:
            raise ValueError('Error %d while encrypting in OFB mode' % result)
        if output is None:
            return get_raw_buffer(ciphertext)
        else:
            return None

    def decrypt(self, ciphertext, output=None):
        """Decrypt data with the key and the parameters set at initialization.

        A cipher object is stateful: once you have decrypted a message
        you cannot decrypt (or encrypt) another message with the same
        object.

        The data to decrypt can be broken up in two or
        more pieces and `decrypt` can be called multiple times.

        That is, the statement:

            >>> c.decrypt(a) + c.decrypt(b)

        is equivalent to:

             >>> c.decrypt(a+b)

        This function does not remove any padding from the plaintext.

        :Parameters:
          ciphertext : bytes/bytearray/memoryview
            The piece of data to decrypt.
            It can be of any length.
        :Keywords:
          output : bytearray/memoryview
            The location where the plaintext is written to.
            If ``None``, the plaintext is returned.
        :Return:
          If ``output`` is ``None``, the plaintext is returned as ``bytes``.
          Otherwise, ``None``.
        """
        if 'decrypt' not in self._next:
            raise TypeError('decrypt() cannot be called after encrypt()')
        self._next = ['decrypt']
        if output is None:
            plaintext = create_string_buffer(len(ciphertext))
        else:
            plaintext = output
            if not is_writeable_buffer(output):
                raise TypeError('output must be a bytearray or a writeable memoryview')
            if len(ciphertext) != len(output):
                raise ValueError('output must have the same length as the input  (%d bytes)' % len(plaintext))
        result = raw_ofb_lib.OFB_decrypt(self._state.get(), c_uint8_ptr(ciphertext), c_uint8_ptr(plaintext), c_size_t(len(ciphertext)))
        if result:
            raise ValueError('Error %d while decrypting in OFB mode' % result)
        if output is None:
            return get_raw_buffer(plaintext)
        else:
            return None