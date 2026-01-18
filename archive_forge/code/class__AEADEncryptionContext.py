from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
class _AEADEncryptionContext(_AEADCipherContext, AEADEncryptionContext):

    @property
    def tag(self) -> bytes:
        if self._ctx is not None:
            raise NotYetFinalized('You must finalize encryption before getting the tag.')
        assert self._tag is not None
        return self._tag