import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def _sign_rsa(hash_algorithm_name: str, sig_base_str: str, rsa_private_key: str):
    """
    Calculate the signature for an RSA-based signature method.

    The ``alg`` is used to calculate the digest over the signature base string.
    For the "RSA_SHA1" signature method, the alg must be SHA-1. While OAuth 1.0a
    only defines the RSA-SHA1 signature method, this function can be used for
    other non-standard signature methods that only differ from RSA-SHA1 by the
    digest algorithm.

    Signing for the RSA-SHA1 signature method is defined in
    `section 3.4.3`_ of RFC 5849.

    The RSASSA-PKCS1-v1_5 signature algorithm used defined by
    `RFC3447, Section 8.2`_ (also known as PKCS#1), with the `alg` as the
    hash function for EMSA-PKCS1-v1_5.  To
    use this method, the client MUST have established client credentials
    with the server that included its RSA public key (in a manner that is
    beyond the scope of this specification).

    .. _`section 3.4.3`: https://tools.ietf.org/html/rfc5849#section-3.4.3
    .. _`RFC3447, Section 8.2`: https://tools.ietf.org/html/rfc3447#section-8.2
    """
    alg = _get_jwt_rsa_algorithm(hash_algorithm_name)
    if not rsa_private_key:
        raise ValueError('rsa_private_key required for RSA with ' + alg.hash_alg.name + ' signature method')
    m = sig_base_str.encode('ascii')
    key = _prepare_key_plus(alg, rsa_private_key)
    s = alg.sign(m, key)
    return binascii.b2a_base64(s)[:-1].decode('ascii')