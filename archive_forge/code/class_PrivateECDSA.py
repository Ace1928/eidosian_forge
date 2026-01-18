from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PrivateECDSA(CryptographyPrivateKey):
    key: ec.EllipticCurvePrivateKey
    key_cls = ec.EllipticCurvePrivateKey
    public_cls = PublicECDSA

    def sign(self, data: bytes, verify: bool=False) -> bytes:
        """Sign using a private key per RFC 6605, section 4."""
        der_signature = self.key.sign(data, ec.ECDSA(self.public_cls.chosen_hash))
        dsa_r, dsa_s = utils.decode_dss_signature(der_signature)
        signature = int.to_bytes(dsa_r, length=self.public_cls.octets, byteorder='big') + int.to_bytes(dsa_s, length=self.public_cls.octets, byteorder='big')
        if verify:
            self.public_key().verify(signature, data)
        return signature

    @classmethod
    def generate(cls) -> 'PrivateECDSA':
        return cls(key=ec.generate_private_key(curve=cls.public_cls.curve, backend=default_backend()))