from binascii import unhexlify
from Cryptodome.Cipher import ChaCha20
from Cryptodome.Cipher.ChaCha20 import _HChaCha20
from Cryptodome.Hash import Poly1305, BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Util.py3compat import _copy_bytes, bord
from Cryptodome.Util._raw_api import is_buffer
class ChaCha20Poly1305Cipher(object):
    """ChaCha20-Poly1305 and XChaCha20-Poly1305 cipher object.
    Do not create it directly. Use :py:func:`new` instead.

    :var nonce: The nonce with length 8, 12 or 24 bytes
    :vartype nonce: byte string
    """

    def __init__(self, key, nonce):
        """Initialize a ChaCha20-Poly1305 AEAD cipher object

        See also `new()` at the module level."""
        self._next = ('update', 'encrypt', 'decrypt', 'digest', 'verify')
        self._authenticator = Poly1305.new(key=key, nonce=nonce, cipher=ChaCha20)
        self._cipher = ChaCha20.new(key=key, nonce=nonce)
        self._cipher.seek(64)
        self._len_aad = 0
        self._len_ct = 0
        self._mac_tag = None
        self._status = _CipherStatus.PROCESSING_AUTH_DATA

    def update(self, data):
        """Protect the associated data.

        Associated data (also known as *additional authenticated data* - AAD)
        is the piece of the message that must stay in the clear, while
        still allowing the receiver to verify its integrity.
        An example is packet headers.

        The associated data (possibly split into multiple segments) is
        fed into :meth:`update` before any call to :meth:`decrypt` or :meth:`encrypt`.
        If there is no associated data, :meth:`update` is not called.

        :param bytes/bytearray/memoryview assoc_data:
            A piece of associated data. There are no restrictions on its size.
        """
        if 'update' not in self._next:
            raise TypeError('update() method cannot be called')
        self._len_aad += len(data)
        self._authenticator.update(data)

    def _pad_aad(self):
        assert self._status == _CipherStatus.PROCESSING_AUTH_DATA
        if self._len_aad & 15:
            self._authenticator.update(b'\x00' * (16 - (self._len_aad & 15)))
        self._status = _CipherStatus.PROCESSING_CIPHERTEXT

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
            raise TypeError('encrypt() method cannot be called')
        if self._status == _CipherStatus.PROCESSING_AUTH_DATA:
            self._pad_aad()
        self._next = ('encrypt', 'digest')
        result = self._cipher.encrypt(plaintext, output=output)
        self._len_ct += len(plaintext)
        if output is None:
            self._authenticator.update(result)
        else:
            self._authenticator.update(output)
        return result

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
            raise TypeError('decrypt() method cannot be called')
        if self._status == _CipherStatus.PROCESSING_AUTH_DATA:
            self._pad_aad()
        self._next = ('decrypt', 'verify')
        self._len_ct += len(ciphertext)
        self._authenticator.update(ciphertext)
        return self._cipher.decrypt(ciphertext, output=output)

    def _compute_mac(self):
        """Finalize the cipher (if not done already) and return the MAC."""
        if self._mac_tag:
            assert self._status == _CipherStatus.PROCESSING_DONE
            return self._mac_tag
        assert self._status != _CipherStatus.PROCESSING_DONE
        if self._status == _CipherStatus.PROCESSING_AUTH_DATA:
            self._pad_aad()
        if self._len_ct & 15:
            self._authenticator.update(b'\x00' * (16 - (self._len_ct & 15)))
        self._status = _CipherStatus.PROCESSING_DONE
        self._authenticator.update(long_to_bytes(self._len_aad, 8)[::-1])
        self._authenticator.update(long_to_bytes(self._len_ct, 8)[::-1])
        self._mac_tag = self._authenticator.digest()
        return self._mac_tag

    def digest(self):
        """Compute the *binary* authentication tag (MAC).

        :Return: the MAC tag, as 16 ``bytes``.
        """
        if 'digest' not in self._next:
            raise TypeError('digest() method cannot be called')
        self._next = ('digest',)
        return self._compute_mac()

    def hexdigest(self):
        """Compute the *printable* authentication tag (MAC).

        This method is like :meth:`digest`.

        :Return: the MAC tag, as a hexadecimal string.
        """
        return ''.join(['%02x' % bord(x) for x in self.digest()])

    def verify(self, received_mac_tag):
        """Validate the *binary* authentication tag (MAC).

        The receiver invokes this method at the very end, to
        check if the associated data (if any) and the decrypted
        messages are valid.

        :param bytes/bytearray/memoryview received_mac_tag:
            This is the 16-byte *binary* MAC, as received from the sender.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """
        if 'verify' not in self._next:
            raise TypeError('verify() cannot be called when encrypting a message')
        self._next = ('verify',)
        secret = get_random_bytes(16)
        self._compute_mac()
        mac1 = BLAKE2s.new(digest_bits=160, key=secret, data=self._mac_tag)
        mac2 = BLAKE2s.new(digest_bits=160, key=secret, data=received_mac_tag)
        if mac1.digest() != mac2.digest():
            raise ValueError('MAC check failed')

    def hexverify(self, hex_mac_tag):
        """Validate the *printable* authentication tag (MAC).

        This method is like :meth:`verify`.

        :param string hex_mac_tag:
            This is the *printable* MAC.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """
        self.verify(unhexlify(hex_mac_tag))

    def encrypt_and_digest(self, plaintext):
        """Perform :meth:`encrypt` and :meth:`digest` in one step.

        :param plaintext: The data to encrypt, of any size.
        :type plaintext: bytes/bytearray/memoryview
        :return: a tuple with two ``bytes`` objects:

            - the ciphertext, of equal length as the plaintext
            - the 16-byte MAC tag
        """
        return (self.encrypt(plaintext), self.digest())

    def decrypt_and_verify(self, ciphertext, received_mac_tag):
        """Perform :meth:`decrypt` and :meth:`verify` in one step.

        :param ciphertext: The piece of data to decrypt.
        :type ciphertext: bytes/bytearray/memoryview
        :param bytes received_mac_tag:
            This is the 16-byte *binary* MAC, as received from the sender.
        :return: the decrypted data (as ``bytes``)
        :raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """
        plaintext = self.decrypt(ciphertext)
        self.verify(received_mac_tag)
        return plaintext