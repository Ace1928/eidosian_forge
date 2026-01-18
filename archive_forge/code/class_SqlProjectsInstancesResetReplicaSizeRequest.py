from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlProjectsInstancesResetReplicaSizeRequest(_messages.Message):
    """A SqlProjectsInstancesResetReplicaSizeRequest object.

  Fields:
    instance: Cloud SQL read replica instance name.
    project: ID of the project that contains the read replica.
    sqlInstancesResetReplicaSizeRequest: A SqlInstancesResetReplicaSizeRequest
      resource to be passed as the request body.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    sqlInstancesResetReplicaSizeRequest = _messages.MessageField('SqlInstancesResetReplicaSizeRequest', 3)