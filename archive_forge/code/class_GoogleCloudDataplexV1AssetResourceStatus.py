from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AssetResourceStatus(_messages.Message):
    """Status of the resource referenced by an asset.

  Enums:
    StateValueValuesEnum: The current state of the managed resource.

  Fields:
    managedAccessIdentity: Output only. Service account associated with the
      BigQuery Connection.
    message: Additional information about the current state.
    state: The current state of the managed resource.
    updateTime: Last update time of the status.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the managed resource.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      READY: Resource does not have any errors.
      ERROR: Resource has errors.
    """
        STATE_UNSPECIFIED = 0
        READY = 1
        ERROR = 2
    managedAccessIdentity = _messages.StringField(1)
    message = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    updateTime = _messages.StringField(4)