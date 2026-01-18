from __future__ import absolute_import
import sqlalchemy.types
from oauth2client import client
class CredentialsType(sqlalchemy.types.PickleType):
    """Type representing credentials.

    Alias for :class:`sqlalchemy.types.PickleType`.
    """