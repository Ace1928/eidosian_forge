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
def _canonicalize_hostname(hostname: str) -> str:
    """Canonicalize hostname following MIT-krb5 behavior."""
    af, socktype, proto, canonname, sockaddr = socket.getaddrinfo(hostname, None, 0, 0, socket.IPPROTO_TCP, socket.AI_CANONNAME)[0]
    try:
        name = socket.getnameinfo(sockaddr, socket.NI_NAMEREQD)
    except socket.gaierror:
        return canonname.lower()
    return name[0].lower()