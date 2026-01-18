from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LastEditorValueValuesEnum(_messages.Enum):
    """Output only. Output Only. Indicates who most recently edited the
    upgrade schedule. The value is updated whenever the upgrade is
    rescheduled.

    Values:
      EDITOR_UNSPECIFIED: The default value. This value should never be used.
      SYSTEM: The upgrade is scheduled by the System or internal service.
      USER: The upgrade is scheduled by the end user.
    """
    EDITOR_UNSPECIFIED = 0
    SYSTEM = 1
    USER = 2