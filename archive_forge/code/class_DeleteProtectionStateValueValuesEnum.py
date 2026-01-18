from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeleteProtectionStateValueValuesEnum(_messages.Enum):
    """State of delete protection for the database.

    Values:
      DELETE_PROTECTION_STATE_UNSPECIFIED: The default value. Delete
        protection type is not specified
      DELETE_PROTECTION_DISABLED: Delete protection is disabled
      DELETE_PROTECTION_ENABLED: Delete protection is enabled
    """
    DELETE_PROTECTION_STATE_UNSPECIFIED = 0
    DELETE_PROTECTION_DISABLED = 1
    DELETE_PROTECTION_ENABLED = 2