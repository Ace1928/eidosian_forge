from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PrivateECDSAP256SHA256(PrivateECDSA):
    public_cls = PublicECDSAP256SHA256