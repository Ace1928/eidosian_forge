from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetailOfMessageValueValuesEnum(_messages.Enum):
    """How much data to include in the Pub/Sub message. If the user wishes to
    limit the size of the message, they can use resource_name and fetch the
    profile fields they wish to. Per table profile (not per column).

    Values:
      DETAIL_LEVEL_UNSPECIFIED: Unused.
      TABLE_PROFILE: The full table data profile.
      RESOURCE_NAME: The resource name of the table.
    """
    DETAIL_LEVEL_UNSPECIFIED = 0
    TABLE_PROFILE = 1
    RESOURCE_NAME = 2