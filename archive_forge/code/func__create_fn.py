from __future__ import annotations
import os
import typing
from cryptography import exceptions, utils
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.bindings._rust import FixedPool
def _create_fn(self):
    return aead._aead_create_ctx(backend, self, self._key)