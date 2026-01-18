from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _mark_asn1_named_ec_curve(backend: Backend, ec_cdata):
    """
    Set the named curve flag on the EC_KEY. This causes OpenSSL to
    serialize EC keys along with their curve OID which makes
    deserialization easier.
    """
    backend._lib.EC_KEY_set_asn1_flag(ec_cdata, backend._lib.OPENSSL_EC_NAMED_CURVE)