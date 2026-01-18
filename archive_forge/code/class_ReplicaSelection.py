from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicaSelection(_messages.Message):
    """The directed read replica selector. Callers must provide one or more of
  the following fields for replica selection: * `location` - The location must
  be one of the regions within the multi-region configuration of your
  database. * `type` - The type of the replica. Some examples of using
  replica_selectors are: * `location:us-east1` --> The "us-east1" replica(s)
  of any available type will be used to process the request. *
  `type:READ_ONLY` --> The "READ_ONLY" type replica(s) in nearest available
  location will be used to process the request. * `location:us-east1
  type:READ_ONLY` --> The "READ_ONLY" type replica(s) in location "us-east1"
  will be used to process the request.

  Enums:
    TypeValueValuesEnum: The type of replica.

  Fields:
    location: The location or region of the serving requests, e.g. "us-east1".
    type: The type of replica.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of replica.

    Values:
      TYPE_UNSPECIFIED: Not specified.
      READ_WRITE: Read-write replicas support both reads and writes.
      READ_ONLY: Read-only replicas only support reads (not writes).
    """
        TYPE_UNSPECIFIED = 0
        READ_WRITE = 1
        READ_ONLY = 2
    location = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)