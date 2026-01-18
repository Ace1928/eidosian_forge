from __future__ import annotations
import collections
import contextlib
import itertools
import typing
from contextlib import contextmanager
from cryptography import utils, x509
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.ciphers import _CipherContext
from cryptography.hazmat.backends.openssl.cmac import _CMACContext
from cryptography.hazmat.backends.openssl.ec import (
from cryptography.hazmat.backends.openssl.rsa import (
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.bindings.openssl import binding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.ciphers.algorithms import (
from cryptography.hazmat.primitives.ciphers.modes import (
from cryptography.hazmat.primitives.serialization import ssh
from cryptography.hazmat.primitives.serialization.pkcs12 import (
def _register_default_ciphers(self) -> None:
    for cipher_cls in [AES, AES128, AES256]:
        for mode_cls in [CBC, CTR, ECB, OFB, CFB, CFB8, GCM]:
            self.register_cipher_adapter(cipher_cls, mode_cls, GetCipherByName('{cipher.name}-{cipher.key_size}-{mode.name}'))
    for mode_cls in [CBC, CTR, ECB, OFB, CFB]:
        self.register_cipher_adapter(Camellia, mode_cls, GetCipherByName('{cipher.name}-{cipher.key_size}-{mode.name}'))
    for mode_cls in [CBC, CFB, CFB8, OFB]:
        self.register_cipher_adapter(TripleDES, mode_cls, GetCipherByName('des-ede3-{mode.name}'))
    self.register_cipher_adapter(TripleDES, ECB, GetCipherByName('des-ede3'))
    self.register_cipher_adapter(ChaCha20, type(None), GetCipherByName('chacha20'))
    self.register_cipher_adapter(AES, XTS, _get_xts_cipher)
    for mode_cls in [ECB, CBC, OFB, CFB, CTR]:
        self.register_cipher_adapter(SM4, mode_cls, GetCipherByName('sm4-{mode.name}'))
    if self._binding._legacy_provider_loaded or not self._lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER:
        for mode_cls in [CBC, CFB, OFB, ECB]:
            self.register_cipher_adapter(_BlowfishInternal, mode_cls, GetCipherByName('bf-{mode.name}'))
        for mode_cls in [CBC, CFB, OFB, ECB]:
            self.register_cipher_adapter(_SEEDInternal, mode_cls, GetCipherByName('seed-{mode.name}'))
        for cipher_cls, mode_cls in itertools.product([_CAST5Internal, _IDEAInternal], [CBC, OFB, CFB, ECB]):
            self.register_cipher_adapter(cipher_cls, mode_cls, GetCipherByName('{cipher.name}-{mode.name}'))
        self.register_cipher_adapter(ARC4, type(None), GetCipherByName('rc4'))
        self.register_cipher_adapter(_RC2, type(None), GetCipherByName('rc2'))