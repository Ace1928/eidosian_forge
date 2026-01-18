from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlServerUserDetails(_messages.Message):
    """Represents a Sql Server user on the Cloud SQL instance.

  Fields:
    disabled: If the user has been disabled
    serverRoles: The server roles for this user
  """
    disabled = _messages.BooleanField(1)
    serverRoles = _messages.StringField(2, repeated=True)