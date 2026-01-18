from typing import Type
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed448, ed25519
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PublicED448(PublicEDDSA):
    key: ed448.Ed448PublicKey
    key_cls = ed448.Ed448PublicKey
    algorithm = Algorithm.ED448