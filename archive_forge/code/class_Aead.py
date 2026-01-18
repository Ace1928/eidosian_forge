from typing import ClassVar, Optional
import nacl.bindings
from nacl import encoding
from nacl import exceptions as exc
from nacl.utils import EncryptedMessage, StringFixer, random
class Aead(encoding.Encodable, StringFixer):
    """
    The AEAD class encrypts and decrypts messages using the given secret key.

    Unlike :class:`~nacl.secret.SecretBox`, AEAD supports authenticating
    non-confidential data received alongside the message, such as a length
    or type tag.

    Like :class:`~nacl.secret.Secretbox`, this class provides authenticated
    encryption. An inauthentic message will cause the decrypt function to raise
    an exception.

    Likewise, the authenticator should not be mistaken for a (public-key)
    signature: recipients (with the ability to decrypt messages) are capable of
    creating arbitrary valid message; in particular, this means AEAD messages
    are repudiable. For non-repudiable messages, sign them after encryption.

    The cryptosystem used is `XChacha20-Poly1305`_ as specified for
    `standardization`_. There are `no practical limits`_ to how much can safely
    be encrypted under a given key (up to 2⁶⁴ messages each containing up
    to 2⁶⁴ bytes).

    .. _standardization: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha
    .. _XChacha20-Poly1305: https://doc.libsodium.org/secret-key_cryptography/aead#xchacha-20-poly1305
    .. _no practical limits: https://doc.libsodium.org/secret-key_cryptography/aead#limitations

    :param key: The secret key used to encrypt and decrypt messages
    :param encoder: The encoder class used to decode the given key

    :cvar KEY_SIZE: The size that the key is required to be.
    :cvar NONCE_SIZE: The size that the nonce is required to be.
    :cvar MACBYTES: The size of the authentication MAC tag in bytes.
    :cvar MESSAGEBYTES_MAX: The maximum size of a message which can be
                            safely encrypted with a single key/nonce
                            pair.
    """
    KEY_SIZE = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_KEYBYTES
    NONCE_SIZE = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
    MACBYTES = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_ABYTES
    MESSAGEBYTES_MAX = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_MESSAGEBYTES_MAX

    def __init__(self, key: bytes, encoder: encoding.Encoder=encoding.RawEncoder):
        key = encoder.decode(key)
        if not isinstance(key, bytes):
            raise exc.TypeError('AEAD must be created from 32 bytes')
        if len(key) != self.KEY_SIZE:
            raise exc.ValueError('The key must be exactly %s bytes long' % self.KEY_SIZE)
        self._key = key

    def __bytes__(self) -> bytes:
        return self._key

    def encrypt(self, plaintext: bytes, aad: bytes=b'', nonce: Optional[bytes]=None, encoder: encoding.Encoder=encoding.RawEncoder) -> EncryptedMessage:
        """
        Encrypts the plaintext message using the given `nonce` (or generates
        one randomly if omitted) and returns the ciphertext encoded with the
        encoder.

        .. warning:: It is vitally important for :param nonce: to be unique.
            By default, it is generated randomly; [:class:`Aead`] uses XChacha20
            for extended (192b) nonce size, so the risk of reusing random nonces
            is negligible.  It is *strongly recommended* to keep this behaviour,
            as nonce reuse will compromise the privacy of encrypted messages.
            Should implicit nonces be inadequate for your application, the
            second best option is using split counters; e.g. if sending messages
            encrypted under a shared key between 2 users, each user can use the
            number of messages it sent so far, prefixed or suffixed with a 1bit
            user id.  Note that the counter must **never** be rolled back (due
            to overflow, on-disk state being rolled back to an earlier backup,
            ...)

        :param plaintext: [:class:`bytes`] The plaintext message to encrypt
        :param nonce: [:class:`bytes`] The nonce to use in the encryption
        :param encoder: The encoder to use to encode the ciphertext
        :rtype: [:class:`nacl.utils.EncryptedMessage`]
        """
        if nonce is None:
            nonce = random(self.NONCE_SIZE)
        if len(nonce) != self.NONCE_SIZE:
            raise exc.ValueError('The nonce must be exactly %s bytes long' % self.NONCE_SIZE)
        ciphertext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(plaintext, aad, nonce, self._key)
        encoded_nonce = encoder.encode(nonce)
        encoded_ciphertext = encoder.encode(ciphertext)
        return EncryptedMessage._from_parts(encoded_nonce, encoded_ciphertext, encoder.encode(nonce + ciphertext))

    def decrypt(self, ciphertext: bytes, aad: bytes=b'', nonce: Optional[bytes]=None, encoder: encoding.Encoder=encoding.RawEncoder) -> bytes:
        """
        Decrypts the ciphertext using the `nonce` (explicitly, when passed as a
        parameter or implicitly, when omitted, as part of the ciphertext) and
        returns the plaintext message.

        :param ciphertext: [:class:`bytes`] The encrypted message to decrypt
        :param nonce: [:class:`bytes`] The nonce used when encrypting the
            ciphertext
        :param encoder: The encoder used to decode the ciphertext.
        :rtype: [:class:`bytes`]
        """
        ciphertext = encoder.decode(ciphertext)
        if nonce is None:
            nonce = ciphertext[:self.NONCE_SIZE]
            ciphertext = ciphertext[self.NONCE_SIZE:]
        if len(nonce) != self.NONCE_SIZE:
            raise exc.ValueError('The nonce must be exactly %s bytes long' % self.NONCE_SIZE)
        plaintext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(ciphertext, aad, nonce, self._key)
        return plaintext