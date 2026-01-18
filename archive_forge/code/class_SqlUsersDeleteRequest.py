from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlUsersDeleteRequest(_messages.Message):
    """A SqlUsersDeleteRequest object.

  Fields:
    host: Host of the user in the instance.
    instance: Database instance ID. This does not include the project ID.
    name: Name of the user in the instance.
    project: Project ID of the project that contains the instance.
  """
    host = _messages.StringField(1)
    instance = _messages.StringField(2, required=True)
    name = _messages.StringField(3)
    project = _messages.StringField(4, required=True)