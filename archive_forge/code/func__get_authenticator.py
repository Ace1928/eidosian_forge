from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def _get_authenticator(credentials: MongoCredential, address: tuple[str, int]) -> _OIDCAuthenticator:
    if credentials.cache.data:
        return credentials.cache.data
    principal_name = credentials.username
    properties = credentials.mechanism_properties
    if not properties.provider_name:
        found = False
        allowed_hosts = properties.allowed_hosts
        for patt in allowed_hosts:
            if patt == address[0]:
                found = True
            elif patt.startswith('*.') and address[0].endswith(patt[1:]):
                found = True
        if not found:
            raise ConfigurationError(f'Refusing to connect to {address[0]}, which is not in authOIDCAllowedHosts: {allowed_hosts}')
    credentials.cache.data = _OIDCAuthenticator(username=principal_name, properties=properties)
    return credentials.cache.data