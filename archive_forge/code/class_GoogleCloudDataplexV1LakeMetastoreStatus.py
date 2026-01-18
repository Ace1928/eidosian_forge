from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1LakeMetastoreStatus(_messages.Message):
    """Status of Lake and Dataproc Metastore service instance association.

  Enums:
    StateValueValuesEnum: Current state of association.

  Fields:
    endpoint: The URI of the endpoint used to access the Metastore service.
    message: Additional information about the current status.
    state: Current state of association.
    updateTime: Last update time of the metastore status of the lake.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Current state of association.

    Values:
      STATE_UNSPECIFIED: Unspecified.
      NONE: A Metastore service instance is not associated with the lake.
      READY: A Metastore service instance is attached to the lake.
      UPDATING: Attach/detach is in progress.
      ERROR: Attach/detach could not be done due to errors.
    """
        STATE_UNSPECIFIED = 0
        NONE = 1
        READY = 2
        UPDATING = 3
        ERROR = 4
    endpoint = _messages.StringField(1)
    message = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    updateTime = _messages.StringField(4)