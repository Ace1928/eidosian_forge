from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudPolicytroubleshooterV1betaBindingExplanationAnnotatedMembership(_messages.Message):
    """Details about whether the binding includes the member.

  Enums:
    MembershipValueValuesEnum: Indicates whether the binding includes the
      member.
    RelevanceValueValuesEnum: The relevance of the member's status to the
      overall determination for the binding.

  Fields:
    membership: Indicates whether the binding includes the member.
    relevance: The relevance of the member's status to the overall
      determination for the binding.
  """

    class MembershipValueValuesEnum(_messages.Enum):
        """Indicates whether the binding includes the member.

    Values:
      MEMBERSHIP_UNSPECIFIED: Reserved for future use.
      MEMBERSHIP_INCLUDED: The binding includes the member. The member can be
        included directly or indirectly. For example: * A member is included
        directly if that member is listed in the binding. * A member is
        included indirectly if that member is in a Google group or G Suite
        domain that is listed in the binding.
      MEMBERSHIP_NOT_INCLUDED: The binding does not include the member.
      MEMBERSHIP_UNKNOWN_INFO_DENIED: The sender of the request is not allowed
        to access the binding.
      MEMBERSHIP_UNKNOWN_UNSUPPORTED: The member is an unsupported type. Only
        Google Accounts and service accounts are supported.
    """
        MEMBERSHIP_UNSPECIFIED = 0
        MEMBERSHIP_INCLUDED = 1
        MEMBERSHIP_NOT_INCLUDED = 2
        MEMBERSHIP_UNKNOWN_INFO_DENIED = 3
        MEMBERSHIP_UNKNOWN_UNSUPPORTED = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the member's status to the overall determination for
    the binding.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Reserved for future use.
      NORMAL: The data point has a limited effect on the result. Changing the
        data point is unlikely to affect the overall determination.
      HIGH: The data point has a strong effect on the result. Changing the
        data point is likely to affect the overall determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        NORMAL = 1
        HIGH = 2
    membership = _messages.EnumField('MembershipValueValuesEnum', 1)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 2)