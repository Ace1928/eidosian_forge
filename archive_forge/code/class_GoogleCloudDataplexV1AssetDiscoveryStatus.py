from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AssetDiscoveryStatus(_messages.Message):
    """Status of discovery for an asset.

  Enums:
    StateValueValuesEnum: The current status of the discovery feature.

  Fields:
    lastRunDuration: The duration of the last discovery run.
    lastRunTime: The start time of the last discovery run.
    message: Additional information about the current state.
    state: The current status of the discovery feature.
    stats: Data Stats of the asset reported by discovery.
    updateTime: Last update time of the status.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current status of the discovery feature.

    Values:
      STATE_UNSPECIFIED: State is unspecified.
      SCHEDULED: Discovery for the asset is scheduled.
      IN_PROGRESS: Discovery for the asset is running.
      PAUSED: Discovery for the asset is currently paused (e.g. due to a lack
        of available resources). It will be automatically resumed.
      DISABLED: Discovery for the asset is disabled.
    """
        STATE_UNSPECIFIED = 0
        SCHEDULED = 1
        IN_PROGRESS = 2
        PAUSED = 3
        DISABLED = 4
    lastRunDuration = _messages.StringField(1)
    lastRunTime = _messages.StringField(2)
    message = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    stats = _messages.MessageField('GoogleCloudDataplexV1AssetDiscoveryStatusStats', 5)
    updateTime = _messages.StringField(6)