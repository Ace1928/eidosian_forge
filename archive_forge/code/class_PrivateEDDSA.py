from typing import Type
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed448, ed25519
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PrivateEDDSA(CryptographyPrivateKey):
    public_cls: Type[PublicEDDSA]

    def sign(self, data: bytes, verify: bool=False) -> bytes:
        """Sign using a private key per RFC 8080, section 4."""
        signature = self.key.sign(data)
        if verify:
            self.public_key().verify(signature, data)
        return signature

    @classmethod
    def generate(cls) -> 'PrivateEDDSA':
        return cls(key=cls.key_cls.generate())