from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityModeValueValuesEnum(_messages.Enum):
    """Optional. The security mode of the routine, if defined. If not
    defined, the security mode is automatically determined from the routine's
    configuration.

    Values:
      SECURITY_MODE_UNSPECIFIED: The security mode of the routine is
        unspecified.
      DEFINER: The routine is to be executed with the privileges of the user
        who defines it.
      INVOKER: The routine is to be executed with the privileges of the user
        who invokes it.
    """
    SECURITY_MODE_UNSPECIFIED = 0
    DEFINER = 1
    INVOKER = 2