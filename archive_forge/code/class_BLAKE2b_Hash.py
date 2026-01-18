from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, tobytes
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
class BLAKE2b_Hash(object):
    """A BLAKE2b hash object.
    Do not instantiate directly. Use the :func:`new` function.

    :ivar oid: ASN.1 Object ID
    :vartype oid: string

    :ivar block_size: the size in bytes of the internal message block,
                      input to the compression function
    :vartype block_size: integer

    :ivar digest_size: the size in bytes of the resulting hash
    :vartype digest_size: integer
    """
    block_size = 64

    def __init__(self, data, key, digest_bytes, update_after_digest):
        self.digest_size = digest_bytes
        self._update_after_digest = update_after_digest
        self._digest_done = False
        if digest_bytes in (20, 32, 48, 64) and (not key):
            self.oid = '1.3.6.1.4.1.1722.12.2.1.' + str(digest_bytes)
        state = VoidPointer()
        result = _raw_blake2b_lib.blake2b_init(state.address_of(), c_uint8_ptr(key), c_size_t(len(key)), c_size_t(digest_bytes))
        if result:
            raise ValueError('Error %d while instantiating BLAKE2b' % result)
        self._state = SmartPointer(state.get(), _raw_blake2b_lib.blake2b_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (bytes/bytearray/memoryview): The next chunk of the message being hashed.
        """
        if self._digest_done and (not self._update_after_digest):
            raise TypeError("You can only call 'digest' or 'hexdigest' on this object")
        result = _raw_blake2b_lib.blake2b_update(self._state.get(), c_uint8_ptr(data), c_size_t(len(data)))
        if result:
            raise ValueError('Error %d while hashing BLAKE2b data' % result)
        return self

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """
        bfr = create_string_buffer(64)
        result = _raw_blake2b_lib.blake2b_digest(self._state.get(), bfr)
        if result:
            raise ValueError('Error %d while creating BLAKE2b digest' % result)
        self._digest_done = True
        return get_raw_buffer(bfr)[:self.digest_size]

    def hexdigest(self):
        """Return the **printable** digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """
        return ''.join(['%02x' % bord(x) for x in tuple(self.digest())])

    def verify(self, mac_tag):
        """Verify that a given **binary** MAC (computed by another party)
        is valid.

        Args:
          mac_tag (bytes/bytearray/memoryview): the expected MAC of the message.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """
        secret = get_random_bytes(16)
        mac1 = new(digest_bits=160, key=secret, data=mac_tag)
        mac2 = new(digest_bits=160, key=secret, data=self.digest())
        if mac1.digest() != mac2.digest():
            raise ValueError('MAC check failed')

    def hexverify(self, hex_mac_tag):
        """Verify that a given **printable** MAC (computed by another party)
        is valid.

        Args:
            hex_mac_tag (string): the expected MAC of the message, as a hexadecimal string.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """
        self.verify(unhexlify(tobytes(hex_mac_tag)))

    def new(self, **kwargs):
        """Return a new instance of a BLAKE2b hash object.
        See :func:`new`.
        """
        if 'digest_bytes' not in kwargs and 'digest_bits' not in kwargs:
            kwargs['digest_bytes'] = self.digest_size
        return new(**kwargs)