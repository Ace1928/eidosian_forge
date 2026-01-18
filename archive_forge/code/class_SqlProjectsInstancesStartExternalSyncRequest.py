from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlProjectsInstancesStartExternalSyncRequest(_messages.Message):
    """A SqlProjectsInstancesStartExternalSyncRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: ID of the project that contains the instance.
    sqlInstancesStartExternalSyncRequest: A
      SqlInstancesStartExternalSyncRequest resource to be passed as the
      request body.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    sqlInstancesStartExternalSyncRequest = _messages.MessageField('SqlInstancesStartExternalSyncRequest', 3)