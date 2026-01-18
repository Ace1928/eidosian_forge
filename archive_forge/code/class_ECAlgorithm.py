from __future__ import annotations
import hashlib
import hmac
import json
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Union, cast, overload
from .exceptions import InvalidKeyError
from .types import HashlibHash, JWKDict
from .utils import (
class ECAlgorithm(Algorithm):
    """
        Performs signing and verification operations using
        ECDSA and the specified hash function
        """
    SHA256: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA256
    SHA384: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA384
    SHA512: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA512

    def __init__(self, hash_alg: type[hashes.HashAlgorithm]) -> None:
        self.hash_alg = hash_alg

    def prepare_key(self, key: AllowedECKeys | str | bytes) -> AllowedECKeys:
        if isinstance(key, (EllipticCurvePrivateKey, EllipticCurvePublicKey)):
            return key
        if not isinstance(key, (bytes, str)):
            raise TypeError('Expecting a PEM-formatted key.')
        key_bytes = force_bytes(key)
        try:
            if key_bytes.startswith(b'ecdsa-sha2-'):
                crypto_key = load_ssh_public_key(key_bytes)
            else:
                crypto_key = load_pem_public_key(key_bytes)
        except ValueError:
            crypto_key = load_pem_private_key(key_bytes, password=None)
        if not isinstance(crypto_key, (EllipticCurvePrivateKey, EllipticCurvePublicKey)):
            raise InvalidKeyError('Expecting a EllipticCurvePrivateKey/EllipticCurvePublicKey. Wrong key provided for ECDSA algorithms')
        return crypto_key

    def sign(self, msg: bytes, key: EllipticCurvePrivateKey) -> bytes:
        der_sig = key.sign(msg, ECDSA(self.hash_alg()))
        return der_to_raw_signature(der_sig, key.curve)

    def verify(self, msg: bytes, key: 'AllowedECKeys', sig: bytes) -> bool:
        try:
            der_sig = raw_to_der_signature(sig, key.curve)
        except ValueError:
            return False
        try:
            public_key = key.public_key() if isinstance(key, EllipticCurvePrivateKey) else key
            public_key.verify(der_sig, msg, ECDSA(self.hash_alg()))
            return True
        except InvalidSignature:
            return False

    @overload
    @staticmethod
    def to_jwk(key_obj: AllowedECKeys, as_dict: Literal[True]) -> JWKDict:
        ...

    @overload
    @staticmethod
    def to_jwk(key_obj: AllowedECKeys, as_dict: Literal[False]=False) -> str:
        ...

    @staticmethod
    def to_jwk(key_obj: AllowedECKeys, as_dict: bool=False) -> Union[JWKDict, str]:
        if isinstance(key_obj, EllipticCurvePrivateKey):
            public_numbers = key_obj.public_key().public_numbers()
        elif isinstance(key_obj, EllipticCurvePublicKey):
            public_numbers = key_obj.public_numbers()
        else:
            raise InvalidKeyError('Not a public or private key')
        if isinstance(key_obj.curve, SECP256R1):
            crv = 'P-256'
        elif isinstance(key_obj.curve, SECP384R1):
            crv = 'P-384'
        elif isinstance(key_obj.curve, SECP521R1):
            crv = 'P-521'
        elif isinstance(key_obj.curve, SECP256K1):
            crv = 'secp256k1'
        else:
            raise InvalidKeyError(f'Invalid curve: {key_obj.curve}')
        obj: dict[str, Any] = {'kty': 'EC', 'crv': crv, 'x': to_base64url_uint(public_numbers.x).decode(), 'y': to_base64url_uint(public_numbers.y).decode()}
        if isinstance(key_obj, EllipticCurvePrivateKey):
            obj['d'] = to_base64url_uint(key_obj.private_numbers().private_value).decode()
        if as_dict:
            return obj
        else:
            return json.dumps(obj)

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> AllowedECKeys:
        try:
            if isinstance(jwk, str):
                obj = json.loads(jwk)
            elif isinstance(jwk, dict):
                obj = jwk
            else:
                raise ValueError
        except ValueError:
            raise InvalidKeyError('Key is not valid JSON')
        if obj.get('kty') != 'EC':
            raise InvalidKeyError('Not an Elliptic curve key')
        if 'x' not in obj or 'y' not in obj:
            raise InvalidKeyError('Not an Elliptic curve key')
        x = base64url_decode(obj.get('x'))
        y = base64url_decode(obj.get('y'))
        curve = obj.get('crv')
        curve_obj: EllipticCurve
        if curve == 'P-256':
            if len(x) == len(y) == 32:
                curve_obj = SECP256R1()
            else:
                raise InvalidKeyError('Coords should be 32 bytes for curve P-256')
        elif curve == 'P-384':
            if len(x) == len(y) == 48:
                curve_obj = SECP384R1()
            else:
                raise InvalidKeyError('Coords should be 48 bytes for curve P-384')
        elif curve == 'P-521':
            if len(x) == len(y) == 66:
                curve_obj = SECP521R1()
            else:
                raise InvalidKeyError('Coords should be 66 bytes for curve P-521')
        elif curve == 'secp256k1':
            if len(x) == len(y) == 32:
                curve_obj = SECP256K1()
            else:
                raise InvalidKeyError('Coords should be 32 bytes for curve secp256k1')
        else:
            raise InvalidKeyError(f'Invalid curve: {curve}')
        public_numbers = EllipticCurvePublicNumbers(x=int.from_bytes(x, byteorder='big'), y=int.from_bytes(y, byteorder='big'), curve=curve_obj)
        if 'd' not in obj:
            return public_numbers.public_key()
        d = base64url_decode(obj.get('d'))
        if len(d) != len(x):
            raise InvalidKeyError('D should be {} bytes for curve {}', len(x), curve)
        return EllipticCurvePrivateNumbers(int.from_bytes(d, byteorder='big'), public_numbers).private_key()