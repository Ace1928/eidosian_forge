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
class _DigestAuthChallenge(typing.NamedTuple):
    realm: bytes
    nonce: bytes
    algorithm: str
    opaque: bytes | None
    qop: bytes | None