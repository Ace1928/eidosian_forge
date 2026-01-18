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
class _X509Context(_AuthContext):

    def speculate_command(self) -> MutableMapping[str, Any]:
        cmd = SON([('authenticate', 1), ('mechanism', 'MONGODB-X509')])
        if self.credentials.username is not None:
            cmd['user'] = self.credentials.username
        return cmd