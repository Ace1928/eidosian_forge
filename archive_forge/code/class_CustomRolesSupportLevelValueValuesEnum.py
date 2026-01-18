from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomRolesSupportLevelValueValuesEnum(_messages.Enum):
    """The current custom role support level.

    Values:
      SUPPORTED: Default state. Permission is fully supported for custom role
        use.
      TESTING: Permission is being tested to check custom role compatibility.
      NOT_SUPPORTED: Permission is not supported for custom role use.
    """
    SUPPORTED = 0
    TESTING = 1
    NOT_SUPPORTED = 2