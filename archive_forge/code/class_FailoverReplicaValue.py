from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FailoverReplicaValue(_messages.Message):
    """The name and status of the failover replica.

    Fields:
      available: The availability status of the failover replica. A false
        status indicates that the failover replica is out of sync. The primary
        instance can only failover to the failover replica when the status is
        true.
      name: The name of the failover replica. If specified at instance
        creation, a failover replica is created for the instance. The name
        doesn't include the project ID.
    """
    available = _messages.BooleanField(1)
    name = _messages.StringField(2)