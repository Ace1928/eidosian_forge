from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopStateValueValuesEnum(_messages.Enum):
    """Current stopping state of the instance.

    Values:
      SHUTTING_DOWN: The instance is gracefully shutting down.
      STOPPING: The instance is stopping.
    """
    SHUTTING_DOWN = 0
    STOPPING = 1