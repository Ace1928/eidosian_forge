from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
class _AEADDecryptionContext(_AEADCipherContext, AEADDecryptionContext):

    def finalize_with_tag(self, tag: bytes) -> bytes:
        if self._ctx is None:
            raise AlreadyFinalized('Context was already finalized.')
        data = self._ctx.finalize_with_tag(tag)
        self._tag = self._ctx.tag
        self._ctx = None
        return data