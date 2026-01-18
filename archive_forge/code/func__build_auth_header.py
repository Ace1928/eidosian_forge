from __future__ import annotations
import hashlib
import os
import re
import time
import typing
from base64 import b64encode
from urllib.request import parse_http_list
from ._exceptions import ProtocolError
from ._models import Cookies, Request, Response
from ._utils import to_bytes, to_str, unquote
def _build_auth_header(self, request: Request, challenge: _DigestAuthChallenge) -> str:
    hash_func = self._ALGORITHM_TO_HASH_FUNCTION[challenge.algorithm.upper()]

    def digest(data: bytes) -> bytes:
        return hash_func(data).hexdigest().encode()
    A1 = b':'.join((self._username, challenge.realm, self._password))
    path = request.url.raw_path
    A2 = b':'.join((request.method.encode(), path))
    HA2 = digest(A2)
    nc_value = b'%08x' % self._nonce_count
    cnonce = self._get_client_nonce(self._nonce_count, challenge.nonce)
    self._nonce_count += 1
    HA1 = digest(A1)
    if challenge.algorithm.lower().endswith('-sess'):
        HA1 = digest(b':'.join((HA1, challenge.nonce, cnonce)))
    qop = self._resolve_qop(challenge.qop, request=request)
    if qop is None:
        digest_data = [HA1, challenge.nonce, HA2]
    else:
        digest_data = [HA1, challenge.nonce, nc_value, cnonce, qop, HA2]
    format_args = {'username': self._username, 'realm': challenge.realm, 'nonce': challenge.nonce, 'uri': path, 'response': digest(b':'.join(digest_data)), 'algorithm': challenge.algorithm.encode()}
    if challenge.opaque:
        format_args['opaque'] = challenge.opaque
    if qop:
        format_args['qop'] = b'auth'
        format_args['nc'] = nc_value
        format_args['cnonce'] = cnonce
    return 'Digest ' + self._get_header_value(format_args)