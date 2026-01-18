import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dsa, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PublicDSA(CryptographyPublicKey):
    key: dsa.DSAPublicKey
    key_cls = dsa.DSAPublicKey
    algorithm = Algorithm.DSA
    chosen_hash = hashes.SHA1()

    def verify(self, signature: bytes, data: bytes) -> None:
        sig_r = signature[1:21]
        sig_s = signature[21:]
        sig = utils.encode_dss_signature(int.from_bytes(sig_r, 'big'), int.from_bytes(sig_s, 'big'))
        self.key.verify(sig, data, self.chosen_hash)

    def encode_key_bytes(self) -> bytes:
        """Encode a public key per RFC 2536, section 2."""
        pn = self.key.public_numbers()
        dsa_t = (self.key.key_size // 8 - 64) // 8
        if dsa_t > 8:
            raise ValueError('unsupported DSA key size')
        octets = 64 + dsa_t * 8
        res = struct.pack('!B', dsa_t)
        res += pn.parameter_numbers.q.to_bytes(20, 'big')
        res += pn.parameter_numbers.p.to_bytes(octets, 'big')
        res += pn.parameter_numbers.g.to_bytes(octets, 'big')
        res += pn.y.to_bytes(octets, 'big')
        return res

    @classmethod
    def from_dnskey(cls, key: DNSKEY) -> 'PublicDSA':
        cls._ensure_algorithm_key_combination(key)
        keyptr = key.key
        t, = struct.unpack('!B', keyptr[0:1])
        keyptr = keyptr[1:]
        octets = 64 + t * 8
        dsa_q = keyptr[0:20]
        keyptr = keyptr[20:]
        dsa_p = keyptr[0:octets]
        keyptr = keyptr[octets:]
        dsa_g = keyptr[0:octets]
        keyptr = keyptr[octets:]
        dsa_y = keyptr[0:octets]
        return cls(key=dsa.DSAPublicNumbers(int.from_bytes(dsa_y, 'big'), dsa.DSAParameterNumbers(int.from_bytes(dsa_p, 'big'), int.from_bytes(dsa_q, 'big'), int.from_bytes(dsa_g, 'big'))).public_key(default_backend()))