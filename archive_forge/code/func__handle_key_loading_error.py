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
def _handle_key_loading_error(self) -> typing.NoReturn:
    errors = self._consume_errors()
    if not errors:
        raise ValueError('Could not deserialize key data. The data may be in an incorrect format or it may be encrypted with an unsupported algorithm.')
    elif errors[0]._lib_reason_match(self._lib.ERR_LIB_EVP, self._lib.EVP_R_BAD_DECRYPT) or errors[0]._lib_reason_match(self._lib.ERR_LIB_PKCS12, self._lib.PKCS12_R_PKCS12_CIPHERFINAL_ERROR) or (self._lib.Cryptography_HAS_PROVIDERS and errors[0]._lib_reason_match(self._lib.ERR_LIB_PROV, self._lib.PROV_R_BAD_DECRYPT)):
        raise ValueError('Bad decrypt. Incorrect password?')
    elif any((error._lib_reason_match(self._lib.ERR_LIB_EVP, self._lib.EVP_R_UNSUPPORTED_PRIVATE_KEY_ALGORITHM) for error in errors)):
        raise ValueError('Unsupported public key algorithm.')
    else:
        raise ValueError('Could not deserialize key data. The data may be in an incorrect format, it may be encrypted with an unsupported algorithm, or it may be an unsupported key type (e.g. EC curves with explicit parameters).', errors)