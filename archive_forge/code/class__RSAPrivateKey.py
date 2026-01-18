from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
class _RSAPrivateKey(RSAPrivateKey):
    _evp_pkey: object
    _rsa_cdata: object
    _key_size: int

    def __init__(self, backend: Backend, rsa_cdata, evp_pkey, *, unsafe_skip_rsa_key_validation: bool):
        res: int
        if not unsafe_skip_rsa_key_validation:
            res = backend._lib.RSA_check_key(rsa_cdata)
            if res != 1:
                errors = backend._consume_errors()
                raise ValueError('Invalid private key', errors)
            p = backend._ffi.new('BIGNUM **')
            q = backend._ffi.new('BIGNUM **')
            backend._lib.RSA_get0_factors(rsa_cdata, p, q)
            backend.openssl_assert(p[0] != backend._ffi.NULL)
            backend.openssl_assert(q[0] != backend._ffi.NULL)
            p_odd = backend._lib.BN_is_odd(p[0])
            q_odd = backend._lib.BN_is_odd(q[0])
            if p_odd != 1 or q_odd != 1:
                errors = backend._consume_errors()
                raise ValueError('Invalid private key', errors)
        self._backend = backend
        self._rsa_cdata = rsa_cdata
        self._evp_pkey = evp_pkey
        self._blinded = False
        self._blinding_lock = threading.Lock()
        n = self._backend._ffi.new('BIGNUM **')
        self._backend._lib.RSA_get0_key(self._rsa_cdata, n, self._backend._ffi.NULL, self._backend._ffi.NULL)
        self._backend.openssl_assert(n[0] != self._backend._ffi.NULL)
        self._key_size = self._backend._lib.BN_num_bits(n[0])

    def _enable_blinding(self) -> None:
        if not self._blinded:
            with self._blinding_lock:
                self._non_threadsafe_enable_blinding()

    def _non_threadsafe_enable_blinding(self) -> None:
        if not self._blinded:
            res = self._backend._lib.RSA_blinding_on(self._rsa_cdata, self._backend._ffi.NULL)
            self._backend.openssl_assert(res == 1)
            self._blinded = True

    @property
    def key_size(self) -> int:
        return self._key_size

    def decrypt(self, ciphertext: bytes, padding: AsymmetricPadding) -> bytes:
        self._enable_blinding()
        key_size_bytes = (self.key_size + 7) // 8
        if key_size_bytes != len(ciphertext):
            raise ValueError('Ciphertext length must be equal to key size.')
        return _enc_dec_rsa(self._backend, self, ciphertext, padding)

    def public_key(self) -> RSAPublicKey:
        ctx = self._backend._lib.RSAPublicKey_dup(self._rsa_cdata)
        self._backend.openssl_assert(ctx != self._backend._ffi.NULL)
        ctx = self._backend._ffi.gc(ctx, self._backend._lib.RSA_free)
        evp_pkey = self._backend._rsa_cdata_to_evp_pkey(ctx)
        return _RSAPublicKey(self._backend, ctx, evp_pkey)

    def private_numbers(self) -> RSAPrivateNumbers:
        n = self._backend._ffi.new('BIGNUM **')
        e = self._backend._ffi.new('BIGNUM **')
        d = self._backend._ffi.new('BIGNUM **')
        p = self._backend._ffi.new('BIGNUM **')
        q = self._backend._ffi.new('BIGNUM **')
        dmp1 = self._backend._ffi.new('BIGNUM **')
        dmq1 = self._backend._ffi.new('BIGNUM **')
        iqmp = self._backend._ffi.new('BIGNUM **')
        self._backend._lib.RSA_get0_key(self._rsa_cdata, n, e, d)
        self._backend.openssl_assert(n[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(e[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(d[0] != self._backend._ffi.NULL)
        self._backend._lib.RSA_get0_factors(self._rsa_cdata, p, q)
        self._backend.openssl_assert(p[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(q[0] != self._backend._ffi.NULL)
        self._backend._lib.RSA_get0_crt_params(self._rsa_cdata, dmp1, dmq1, iqmp)
        self._backend.openssl_assert(dmp1[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(dmq1[0] != self._backend._ffi.NULL)
        self._backend.openssl_assert(iqmp[0] != self._backend._ffi.NULL)
        return RSAPrivateNumbers(p=self._backend._bn_to_int(p[0]), q=self._backend._bn_to_int(q[0]), d=self._backend._bn_to_int(d[0]), dmp1=self._backend._bn_to_int(dmp1[0]), dmq1=self._backend._bn_to_int(dmq1[0]), iqmp=self._backend._bn_to_int(iqmp[0]), public_numbers=RSAPublicNumbers(e=self._backend._bn_to_int(e[0]), n=self._backend._bn_to_int(n[0])))

    def private_bytes(self, encoding: serialization.Encoding, format: serialization.PrivateFormat, encryption_algorithm: serialization.KeySerializationEncryption) -> bytes:
        return self._backend._private_key_bytes(encoding, format, encryption_algorithm, self, self._evp_pkey, self._rsa_cdata)

    def sign(self, data: bytes, padding: AsymmetricPadding, algorithm: typing.Union[asym_utils.Prehashed, hashes.HashAlgorithm]) -> bytes:
        self._enable_blinding()
        data, algorithm = _calculate_digest_and_algorithm(data, algorithm)
        return _rsa_sig_sign(self._backend, padding, algorithm, self, data)