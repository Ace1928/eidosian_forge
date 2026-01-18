from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateInitiatorValueValuesEnum(_messages.Enum):
    """Output only. The initiator of the QueuedResources's current state.
    Used to indicate whether the SUSPENDING/SUSPENDED state was initiated by
    the user or the service.

    Values:
      STATE_INITIATOR_UNSPECIFIED: The state initiator is unspecified.
      USER: The current QueuedResource state was initiated by the user.
      SERVICE: The current QueuedResource state was initiated by the service.
    """
    STATE_INITIATOR_UNSPECIFIED = 0
    USER = 1
    SERVICE = 2