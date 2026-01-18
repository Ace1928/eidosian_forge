from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsloginUsersProjectsDeleteRequest(_messages.Message):
    """A OsloginUsersProjectsDeleteRequest object.

  Fields:
    name: Required. A reference to the POSIX account to update. POSIX accounts
      are identified by the project ID they are associated with. A reference
      to the POSIX account is in format `users/{user}/projects/{project}`.
  """
    name = _messages.StringField(1, required=True)