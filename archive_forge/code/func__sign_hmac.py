import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def _sign_hmac(hash_algorithm_name: str, sig_base_str: str, client_secret: str, resource_owner_secret: str):
    """
    **HMAC-SHA256**

    The "HMAC-SHA256" signature method uses the HMAC-SHA256 signature
    algorithm as defined in `RFC4634`_::

        digest = HMAC-SHA256 (key, text)

    Per `section 3.4.2`_ of the spec.

    .. _`RFC4634`: https://tools.ietf.org/html/rfc4634
    .. _`section 3.4.2`: https://tools.ietf.org/html/rfc5849#section-3.4.2
    """
    text = sig_base_str
    key = utils.escape(client_secret or '')
    key += '&'
    key += utils.escape(resource_owner_secret or '')
    m = {'SHA-1': hashlib.sha1, 'SHA-256': hashlib.sha256, 'SHA-512': hashlib.sha512}
    hash_alg = m[hash_algorithm_name]
    key_utf8 = key.encode('utf-8')
    text_utf8 = text.encode('utf-8')
    signature = hmac.new(key_utf8, text_utf8, hash_alg)
    return binascii.b2a_base64(signature.digest())[:-1].decode('utf-8')