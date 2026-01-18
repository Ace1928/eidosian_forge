import struct
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.py3compat import _copy_bytes, is_native_int
from Cryptodome.Util.number import long_to_bytes
class CtrMode(object):
    """*CounTeR (CTR)* mode.

    This mode is very similar to ECB, in that
    encryption of one block is done independently of all other blocks.

    Unlike ECB, the block *position* contributes to the encryption
    and no information leaks about symbol frequency.

    Each message block is associated to a *counter* which
    must be unique across all messages that get encrypted
    with the same key (not just within the same message).
    The counter is as big as the block size.

    Counters can be generated in several ways. The most
    straightword one is to choose an *initial counter block*
    (which can be made public, similarly to the *IV* for the
    other modes) and increment its lowest **m** bits by one
    (modulo *2^m*) for each block. In most cases, **m** is
    chosen to be half the block size.

    See `NIST SP800-38A`_, Section 6.5 (for the mode) and
    Appendix B (for how to manage the *initial counter block*).

    .. _`NIST SP800-38A` : http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf

    :undocumented: __init__
    """

    def __init__(self, block_cipher, initial_counter_block, prefix_len, counter_len, little_endian):
        """Create a new block cipher, configured in CTR mode.

        :Parameters:
          block_cipher : C pointer
            A smart pointer to the low-level block cipher instance.

          initial_counter_block : bytes/bytearray/memoryview
            The initial plaintext to use to generate the key stream.

            It is as large as the cipher block, and it embeds
            the initial value of the counter.

            This value must not be reused.
            It shall contain a nonce or a random component.
            Reusing the *initial counter block* for encryptions
            performed with the same key compromises confidentiality.

          prefix_len : integer
            The amount of bytes at the beginning of the counter block
            that never change.

          counter_len : integer
            The length in bytes of the counter embedded in the counter
            block.

          little_endian : boolean
            True if the counter in the counter block is an integer encoded
            in little endian mode. If False, it is big endian.
        """
        if len(initial_counter_block) == prefix_len + counter_len:
            self.nonce = _copy_bytes(None, prefix_len, initial_counter_block)
            'Nonce; not available if there is a fixed suffix'
        self._state = VoidPointer()
        result = raw_ctr_lib.CTR_start_operation(block_cipher.get(), c_uint8_ptr(initial_counter_block), c_size_t(len(initial_counter_block)), c_size_t(prefix_len), counter_len, little_endian, self._state.address_of())
        if result:
            raise ValueError('Error %X while instantiating the CTR mode' % result)
        self._state = SmartPointer(self._state.get(), raw_ctr_lib.CTR_stop_operation)
        block_cipher.release()
        self.block_size = len(initial_counter_block)
        'The block size of the underlying cipher, in bytes.'
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
        result = raw_ctr_lib.CTR_encrypt(self._state.get(), c_uint8_ptr(plaintext), c_uint8_ptr(ciphertext), c_size_t(len(plaintext)))
        if result:
            if result == 393218:
                raise OverflowError('The counter has wrapped around in CTR mode')
            raise ValueError('Error %X while encrypting in CTR mode' % result)
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
            The location where the plaintext must be written to.
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
        result = raw_ctr_lib.CTR_decrypt(self._state.get(), c_uint8_ptr(ciphertext), c_uint8_ptr(plaintext), c_size_t(len(ciphertext)))
        if result:
            if result == 393218:
                raise OverflowError('The counter has wrapped around in CTR mode')
            raise ValueError('Error %X while decrypting in CTR mode' % result)
        if output is None:
            return get_raw_buffer(plaintext)
        else:
            return None