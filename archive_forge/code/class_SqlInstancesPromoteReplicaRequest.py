from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesPromoteReplicaRequest(_messages.Message):
    """A SqlInstancesPromoteReplicaRequest object.

  Fields:
    failover: Set to true if the promote operation should attempt to re-add
      the original primary as a replica when it comes back online. Otherwise,
      if this value is false or not set, the original primary will be a
      standalone instance.
    instance: Cloud SQL read replica instance name.
    project: ID of the project that contains the read replica.
  """
    failover = _messages.BooleanField(1)
    instance = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)