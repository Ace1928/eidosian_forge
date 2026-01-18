from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def _authenticate_oidc(credentials: MongoCredential, conn: Connection, reauthenticate: bool) -> Optional[Mapping[str, Any]]:
    """Authenticate using MONGODB-OIDC."""
    authenticator = _get_authenticator(credentials, conn.address)
    if reauthenticate:
        return authenticator.reauthenticate(conn)
    else:
        return authenticator.authenticate(conn)