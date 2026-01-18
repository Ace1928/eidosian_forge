from typing import Type
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed448, ed25519
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PublicEDDSA(CryptographyPublicKey):

    def verify(self, signature: bytes, data: bytes) -> None:
        self.key.verify(signature, data)

    def encode_key_bytes(self) -> bytes:
        """Encode a public key per RFC 8080, section 3."""
        return self.key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

    @classmethod
    def from_dnskey(cls, key: DNSKEY) -> 'PublicEDDSA':
        cls._ensure_algorithm_key_combination(key)
        return cls(key=cls.key_cls.from_public_bytes(key.key))