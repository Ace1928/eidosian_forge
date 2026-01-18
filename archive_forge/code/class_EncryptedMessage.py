import os
from typing import SupportsBytes, Type, TypeVar
import nacl.bindings
from nacl import encoding
class EncryptedMessage(bytes):
    """
    A bytes subclass that holds a messaged that has been encrypted by a
    :class:`SecretBox`.
    """
    _nonce: bytes
    _ciphertext: bytes

    @classmethod
    def _from_parts(cls: Type[_EncryptedMessage], nonce: bytes, ciphertext: bytes, combined: bytes) -> _EncryptedMessage:
        obj = cls(combined)
        obj._nonce = nonce
        obj._ciphertext = ciphertext
        return obj

    @property
    def nonce(self) -> bytes:
        """
        The nonce used during the encryption of the :class:`EncryptedMessage`.
        """
        return self._nonce

    @property
    def ciphertext(self) -> bytes:
        """
        The ciphertext contained within the :class:`EncryptedMessage`.
        """
        return self._ciphertext