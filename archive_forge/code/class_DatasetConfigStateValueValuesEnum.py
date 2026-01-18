from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetConfigStateValueValuesEnum(_messages.Enum):
    """Output only. State of the DatasetConfig.

    Values:
      CONFIG_STATE_UNSPECIFIED: Unspecified state.
      CONFIG_STATE_ACTIVE: Active configuration indicates that the
        configuration is actively ingesting data.
      CONFIG_STATE_VERIFICATION_IN_PROGRESS: In this state, the configuration
        is being verified for various permissions.
      CONFIG_STATE_CREATED: Configuration is created and further processing
        needs to happen.
      CONFIG_STATE_PROCESSING: Configuration is under processing
    """
    CONFIG_STATE_UNSPECIFIED = 0
    CONFIG_STATE_ACTIVE = 1
    CONFIG_STATE_VERIFICATION_IN_PROGRESS = 2
    CONFIG_STATE_CREATED = 3
    CONFIG_STATE_PROCESSING = 4