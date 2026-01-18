from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipValueValuesEnum(_messages.Enum):
    """Indicates whether the binding includes the principal.

    Values:
      MEMBERSHIP_UNSPECIFIED: Default value. This value is unused.
      MEMBERSHIP_INCLUDED: The binding includes the principal. The principal
        can be included directly or indirectly. For example: * A principal is
        included directly if that principal is listed in the binding. * A
        principal is included indirectly if that principal is in a Google
        group or Google Workspace domain that is listed in the binding.
      MEMBERSHIP_NOT_INCLUDED: The binding does not include the principal.
      MEMBERSHIP_UNKNOWN_INFO_DENIED: The user who created the Replay is not
        allowed to access the binding.
      MEMBERSHIP_UNKNOWN_UNSUPPORTED: The principal is an unsupported type.
        Only Google Accounts and service accounts are supported.
    """
    MEMBERSHIP_UNSPECIFIED = 0
    MEMBERSHIP_INCLUDED = 1
    MEMBERSHIP_NOT_INCLUDED = 2
    MEMBERSHIP_UNKNOWN_INFO_DENIED = 3
    MEMBERSHIP_UNKNOWN_UNSUPPORTED = 4