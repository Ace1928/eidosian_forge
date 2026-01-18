from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
class SSHCertificate:

    def __init__(self, _nonce: memoryview, _public_key: SSHPublicKeyTypes, _serial: int, _cctype: int, _key_id: memoryview, _valid_principals: typing.List[bytes], _valid_after: int, _valid_before: int, _critical_options: typing.Dict[bytes, bytes], _extensions: typing.Dict[bytes, bytes], _sig_type: memoryview, _sig_key: memoryview, _inner_sig_type: memoryview, _signature: memoryview, _tbs_cert_body: memoryview, _cert_key_type: bytes, _cert_body: memoryview):
        self._nonce = _nonce
        self._public_key = _public_key
        self._serial = _serial
        try:
            self._type = SSHCertificateType(_cctype)
        except ValueError:
            raise ValueError('Invalid certificate type')
        self._key_id = _key_id
        self._valid_principals = _valid_principals
        self._valid_after = _valid_after
        self._valid_before = _valid_before
        self._critical_options = _critical_options
        self._extensions = _extensions
        self._sig_type = _sig_type
        self._sig_key = _sig_key
        self._inner_sig_type = _inner_sig_type
        self._signature = _signature
        self._cert_key_type = _cert_key_type
        self._cert_body = _cert_body
        self._tbs_cert_body = _tbs_cert_body

    @property
    def nonce(self) -> bytes:
        return bytes(self._nonce)

    def public_key(self) -> SSHCertPublicKeyTypes:
        return typing.cast(SSHCertPublicKeyTypes, self._public_key)

    @property
    def serial(self) -> int:
        return self._serial

    @property
    def type(self) -> SSHCertificateType:
        return self._type

    @property
    def key_id(self) -> bytes:
        return bytes(self._key_id)

    @property
    def valid_principals(self) -> typing.List[bytes]:
        return self._valid_principals

    @property
    def valid_before(self) -> int:
        return self._valid_before

    @property
    def valid_after(self) -> int:
        return self._valid_after

    @property
    def critical_options(self) -> typing.Dict[bytes, bytes]:
        return self._critical_options

    @property
    def extensions(self) -> typing.Dict[bytes, bytes]:
        return self._extensions

    def signature_key(self) -> SSHCertPublicKeyTypes:
        sigformat = _lookup_kformat(self._sig_type)
        signature_key, sigkey_rest = sigformat.load_public(self._sig_key)
        _check_empty(sigkey_rest)
        return signature_key

    def public_bytes(self) -> bytes:
        return bytes(self._cert_key_type) + b' ' + binascii.b2a_base64(bytes(self._cert_body), newline=False)

    def verify_cert_signature(self) -> None:
        signature_key = self.signature_key()
        if isinstance(signature_key, ed25519.Ed25519PublicKey):
            signature_key.verify(bytes(self._signature), bytes(self._tbs_cert_body))
        elif isinstance(signature_key, ec.EllipticCurvePublicKey):
            r, data = _get_mpint(self._signature)
            s, data = _get_mpint(data)
            _check_empty(data)
            computed_sig = asym_utils.encode_dss_signature(r, s)
            hash_alg = _get_ec_hash_alg(signature_key.curve)
            signature_key.verify(computed_sig, bytes(self._tbs_cert_body), ec.ECDSA(hash_alg))
        else:
            assert isinstance(signature_key, rsa.RSAPublicKey)
            if self._inner_sig_type == _SSH_RSA:
                hash_alg = hashes.SHA1()
            elif self._inner_sig_type == _SSH_RSA_SHA256:
                hash_alg = hashes.SHA256()
            else:
                assert self._inner_sig_type == _SSH_RSA_SHA512
                hash_alg = hashes.SHA512()
            signature_key.verify(bytes(self._signature), bytes(self._tbs_cert_body), padding.PKCS1v15(), hash_alg)