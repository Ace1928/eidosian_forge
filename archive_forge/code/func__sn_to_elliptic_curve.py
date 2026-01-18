from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _sn_to_elliptic_curve(backend: Backend, sn: str) -> ec.EllipticCurve:
    try:
        return ec._CURVE_TYPES[sn]()
    except KeyError:
        raise UnsupportedAlgorithm(f'{sn} is not a supported elliptic curve', _Reasons.UNSUPPORTED_ELLIPTIC_CURVE)