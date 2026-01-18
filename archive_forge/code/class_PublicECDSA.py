from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PublicECDSA(CryptographyPublicKey):
    key: ec.EllipticCurvePublicKey
    key_cls = ec.EllipticCurvePublicKey
    algorithm: Algorithm
    chosen_hash: hashes.HashAlgorithm
    curve: ec.EllipticCurve
    octets: int

    def verify(self, signature: bytes, data: bytes) -> None:
        sig_r = signature[0:self.octets]
        sig_s = signature[self.octets:]
        sig = utils.encode_dss_signature(int.from_bytes(sig_r, 'big'), int.from_bytes(sig_s, 'big'))
        self.key.verify(sig, data, ec.ECDSA(self.chosen_hash))

    def encode_key_bytes(self) -> bytes:
        """Encode a public key per RFC 6605, section 4."""
        pn = self.key.public_numbers()
        return pn.x.to_bytes(self.octets, 'big') + pn.y.to_bytes(self.octets, 'big')

    @classmethod
    def from_dnskey(cls, key: DNSKEY) -> 'PublicECDSA':
        cls._ensure_algorithm_key_combination(key)
        ecdsa_x = key.key[0:cls.octets]
        ecdsa_y = key.key[cls.octets:cls.octets * 2]
        return cls(key=ec.EllipticCurvePublicNumbers(curve=cls.curve, x=int.from_bytes(ecdsa_x, 'big'), y=int.from_bytes(ecdsa_y, 'big')).public_key(default_backend()))