from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetConnection(_messages.Message):
    """An other-cloud connection to verify before it gets created.

  Fields:
    otherCloudConnection: The content of the connection.
    parent: The parent resource where this connection will be created. It can
      only be an organization number (such as "organizations/123") for now.
      Format: organizations/{organization_number} (e.g.,
      "organizations/123456"). This field is needed when
      non_existent_connection is set. Callers must have
      cloudasset.othercloudconnections.verify permission on the [parent].
  """
    otherCloudConnection = _messages.MessageField('OtherCloudConnection', 1)
    parent = _messages.StringField(2)