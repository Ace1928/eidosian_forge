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
def _parse_challenge(self, request: Request, response: Response, auth_header: str) -> _DigestAuthChallenge:
    """
        Returns a challenge from a Digest WWW-Authenticate header.
        These take the form of:
        `Digest realm="realm@host.com",qop="auth,auth-int",nonce="abc",opaque="xyz"`
        """
    scheme, _, fields = auth_header.partition(' ')
    assert scheme.lower() == 'digest'
    header_dict: dict[str, str] = {}
    for field in parse_http_list(fields):
        key, value = field.strip().split('=', 1)
        header_dict[key] = unquote(value)
    try:
        realm = header_dict['realm'].encode()
        nonce = header_dict['nonce'].encode()
        algorithm = header_dict.get('algorithm', 'MD5')
        opaque = header_dict['opaque'].encode() if 'opaque' in header_dict else None
        qop = header_dict['qop'].encode() if 'qop' in header_dict else None
        return _DigestAuthChallenge(realm=realm, nonce=nonce, algorithm=algorithm, opaque=opaque, qop=qop)
    except KeyError as exc:
        message = 'Malformed Digest WWW-Authenticate header'
        raise ProtocolError(message, request=request) from exc