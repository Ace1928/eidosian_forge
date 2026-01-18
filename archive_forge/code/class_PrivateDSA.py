import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dsa, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PrivateDSA(CryptographyPrivateKey):
    key: dsa.DSAPrivateKey
    key_cls = dsa.DSAPrivateKey
    public_cls = PublicDSA

    def sign(self, data: bytes, verify: bool=False) -> bytes:
        """Sign using a private key per RFC 2536, section 3."""
        public_dsa_key = self.key.public_key()
        if public_dsa_key.key_size > 1024:
            raise ValueError('DSA key size overflow')
        der_signature = self.key.sign(data, self.public_cls.chosen_hash)
        dsa_r, dsa_s = utils.decode_dss_signature(der_signature)
        dsa_t = (public_dsa_key.key_size // 8 - 64) // 8
        octets = 20
        signature = struct.pack('!B', dsa_t) + int.to_bytes(dsa_r, length=octets, byteorder='big') + int.to_bytes(dsa_s, length=octets, byteorder='big')
        if verify:
            self.public_key().verify(signature, data)
        return signature

    @classmethod
    def generate(cls, key_size: int) -> 'PrivateDSA':
        return cls(key=dsa.generate_private_key(key_size=key_size))