from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
class _EllipticCurvePrivateKey(ec.EllipticCurvePrivateKey):

    def __init__(self, backend: Backend, ec_key_cdata, evp_pkey):
        self._backend = backend
        self._ec_key = ec_key_cdata
        self._evp_pkey = evp_pkey
        sn = _ec_key_curve_sn(backend, ec_key_cdata)
        self._curve = _sn_to_elliptic_curve(backend, sn)
        _mark_asn1_named_ec_curve(backend, ec_key_cdata)
        _check_key_infinity(backend, ec_key_cdata)

    @property
    def curve(self) -> ec.EllipticCurve:
        return self._curve

    @property
    def key_size(self) -> int:
        return self.curve.key_size

    def exchange(self, algorithm: ec.ECDH, peer_public_key: ec.EllipticCurvePublicKey) -> bytes:
        if not self._backend.elliptic_curve_exchange_algorithm_supported(algorithm, self.curve):
            raise UnsupportedAlgorithm('This backend does not support the ECDH algorithm.', _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM)
        if peer_public_key.curve.name != self.curve.name:
            raise ValueError('peer_public_key and self are not on the same curve')
        return _evp_pkey_derive(self._backend, self._evp_pkey, peer_public_key)

    def public_key(self) -> ec.EllipticCurvePublicKey:
        group = self._backend._lib.EC_KEY_get0_group(self._ec_key)
        self._backend.openssl_assert(group != self._backend._ffi.NULL)
        curve_nid = self._backend._lib.EC_GROUP_get_curve_name(group)
        public_ec_key = self._backend._ec_key_new_by_curve_nid(curve_nid)
        point = self._backend._lib.EC_KEY_get0_public_key(self._ec_key)
        self._backend.openssl_assert(point != self._backend._ffi.NULL)
        res = self._backend._lib.EC_KEY_set_public_key(public_ec_key, point)
        self._backend.openssl_assert(res == 1)
        evp_pkey = self._backend._ec_cdata_to_evp_pkey(public_ec_key)
        return _EllipticCurvePublicKey(self._backend, public_ec_key, evp_pkey)

    def private_numbers(self) -> ec.EllipticCurvePrivateNumbers:
        bn = self._backend._lib.EC_KEY_get0_private_key(self._ec_key)
        private_value = self._backend._bn_to_int(bn)
        return ec.EllipticCurvePrivateNumbers(private_value=private_value, public_numbers=self.public_key().public_numbers())

    def private_bytes(self, encoding: serialization.Encoding, format: serialization.PrivateFormat, encryption_algorithm: serialization.KeySerializationEncryption) -> bytes:
        return self._backend._private_key_bytes(encoding, format, encryption_algorithm, self, self._evp_pkey, self._ec_key)

    def sign(self, data: bytes, signature_algorithm: ec.EllipticCurveSignatureAlgorithm) -> bytes:
        _check_signature_algorithm(signature_algorithm)
        data, _ = _calculate_digest_and_algorithm(data, signature_algorithm.algorithm)
        return _ecdsa_sig_sign(self._backend, self, data)