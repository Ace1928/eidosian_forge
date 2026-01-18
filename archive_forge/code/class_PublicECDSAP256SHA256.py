from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PublicECDSAP256SHA256(PublicECDSA):
    algorithm = Algorithm.ECDSAP256SHA256
    chosen_hash = hashes.SHA256()
    curve = ec.SECP256R1()
    octets = 32