from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlUsersListRequest(_messages.Message):
    """A SqlUsersListRequest object.

  Fields:
    instance: Database instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)