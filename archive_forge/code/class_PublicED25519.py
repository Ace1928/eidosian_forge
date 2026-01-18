from typing import Type
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed448, ed25519
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PublicED25519(PublicEDDSA):
    key: ed25519.Ed25519PublicKey
    key_cls = ed25519.Ed25519PublicKey
    algorithm = Algorithm.ED25519