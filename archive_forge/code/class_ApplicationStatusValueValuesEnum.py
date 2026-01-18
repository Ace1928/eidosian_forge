from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplicationStatusValueValuesEnum(_messages.Enum):
    """Optional. Search only applications in the chosen state.

    Values:
      APPLICATION_STATUS_UNSPECIFIED: <no description>
      APPLICATION_STATUS_RUNNING: <no description>
      APPLICATION_STATUS_COMPLETED: <no description>
    """
    APPLICATION_STATUS_UNSPECIFIED = 0
    APPLICATION_STATUS_RUNNING = 1
    APPLICATION_STATUS_COMPLETED = 2