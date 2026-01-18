from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
def _padding(self, size: int) -> bytes:
    return bytes([0]) * (size - 1) + bytes([size])