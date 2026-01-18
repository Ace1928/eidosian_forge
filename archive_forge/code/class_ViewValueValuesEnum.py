from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ViewValueValuesEnum(_messages.Enum):
    """Request only the fields for the specified view.

    Values:
      PROJECT_SETTINGS_VIEW_UNSPECIFIED: <no description>
      CONSUMER_VIEW: <no description>
      PRODUCER_VIEW: <no description>
      ALL: <no description>
    """
    PROJECT_SETTINGS_VIEW_UNSPECIFIED = 0
    CONSUMER_VIEW = 1
    PRODUCER_VIEW = 2
    ALL = 3