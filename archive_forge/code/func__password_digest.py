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
def _password_digest(username: str, password: str) -> str:
    """Get a password digest to use for authentication."""
    if not isinstance(password, str):
        raise TypeError('password must be an instance of str')
    if len(password) == 0:
        raise ValueError("password can't be empty")
    if not isinstance(username, str):
        raise TypeError('username must be an instance of str')
    md5hash = hashlib.md5()
    data = f'{username}:mongo:{password}'
    md5hash.update(data.encode('utf-8'))
    return md5hash.hexdigest()