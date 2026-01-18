from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestrictedActionsValueListEntryValuesEnum(_messages.Enum):
    """RestrictedActionsValueListEntryValuesEnum enum type.

    Values:
      RESTRICTED_ACTION_UNSPECIFIED: Unspecified restricted action
      DELETE: Prevent volume from being deleted when mounted.
    """
    RESTRICTED_ACTION_UNSPECIFIED = 0
    DELETE = 1