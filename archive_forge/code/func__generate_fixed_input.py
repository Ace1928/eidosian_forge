from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import (
from cryptography.hazmat.primitives import (
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
def _generate_fixed_input(self) -> bytes:
    if self._fixed_data and isinstance(self._fixed_data, bytes):
        return self._fixed_data
    l_val = utils.int_to_bytes(self._length * 8, self._llen)
    return b''.join([self._label, b'\x00', self._context, l_val])