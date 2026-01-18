from typing import Type
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed448, ed25519
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PrivateED448(PrivateEDDSA):
    key: ed448.Ed448PrivateKey
    key_cls = ed448.Ed448PrivateKey
    public_cls = PublicED448