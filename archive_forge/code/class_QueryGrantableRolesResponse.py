from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class QueryGrantableRolesResponse(_messages.Message):
    """The grantable role query response.

  Fields:
    roles: The list of matching roles.
  """
    roles = _messages.MessageField('Role', 1, repeated=True)