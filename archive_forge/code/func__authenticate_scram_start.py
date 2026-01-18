from __future__ import annotations
import functools
import hashlib
import hmac
import os
import socket
import typing
from base64 import standard_b64decode, standard_b64encode
from collections import namedtuple
from typing import (
from urllib.parse import quote
from bson.binary import Binary
from bson.son import SON
from pymongo.auth_aws import _authenticate_aws
from pymongo.auth_oidc import _authenticate_oidc, _get_authenticator, _OIDCProperties
from pymongo.errors import ConfigurationError, OperationFailure
from pymongo.saslprep import saslprep
def _authenticate_scram_start(credentials: MongoCredential, mechanism: str) -> tuple[bytes, bytes, MutableMapping[str, Any]]:
    username = credentials.username
    user = username.encode('utf-8').replace(b'=', b'=3D').replace(b',', b'=2C')
    nonce = standard_b64encode(os.urandom(32))
    first_bare = b'n=' + user + b',r=' + nonce
    cmd = SON([('saslStart', 1), ('mechanism', mechanism), ('payload', Binary(b'n,,' + first_bare)), ('autoAuthorize', 1), ('options', {'skipEmptyExchange': True})])
    return (nonce, first_bare, cmd)