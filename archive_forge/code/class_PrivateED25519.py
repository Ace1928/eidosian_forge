from typing import Type
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed448, ed25519
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PrivateED25519(PrivateEDDSA):
    key: ed25519.Ed25519PrivateKey
    key_cls = ed25519.Ed25519PrivateKey
    public_cls = PublicED25519