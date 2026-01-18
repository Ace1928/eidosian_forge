from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def _ec_key_curve_sn(backend: Backend, ec_key) -> str:
    group = backend._lib.EC_KEY_get0_group(ec_key)
    backend.openssl_assert(group != backend._ffi.NULL)
    nid = backend._lib.EC_GROUP_get_curve_name(group)
    if nid == backend._lib.NID_undef:
        raise ValueError('ECDSA keys with explicit parameters are unsupported at this time')
    if not backend._lib.CRYPTOGRAPHY_IS_LIBRESSL and backend._lib.EC_GROUP_get_asn1_flag(group) == 0:
        raise ValueError('ECDSA keys with explicit parameters are unsupported at this time')
    curve_name = backend._lib.OBJ_nid2sn(nid)
    backend.openssl_assert(curve_name != backend._ffi.NULL)
    sn = backend._ffi.string(curve_name).decode('ascii')
    return sn