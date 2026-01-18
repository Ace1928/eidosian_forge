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
class _ScramContext(_AuthContext):

    def __init__(self, credentials: MongoCredential, address: tuple[str, int], mechanism: str) -> None:
        super().__init__(credentials, address)
        self.scram_data: Optional[tuple[bytes, bytes]] = None
        self.mechanism = mechanism

    def speculate_command(self) -> Optional[MutableMapping[str, Any]]:
        nonce, first_bare, cmd = _authenticate_scram_start(self.credentials, self.mechanism)
        cmd['db'] = self.credentials.source
        self.scram_data = (nonce, first_bare)
        return cmd