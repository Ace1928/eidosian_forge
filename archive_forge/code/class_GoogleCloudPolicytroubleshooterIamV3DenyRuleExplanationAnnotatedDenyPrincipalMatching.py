from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3DenyRuleExplanationAnnotatedDenyPrincipalMatching(_messages.Message):
    """Details about whether the principal in the request is listed as a denied
  principal in the deny rule, either directly or through membership in a
  principal set.

  Enums:
    MembershipValueValuesEnum: Indicates whether the principal is listed as a
      denied principal in the deny rule, either directly or through membership
      in a principal set.
    RelevanceValueValuesEnum: The relevance of the principal's status to the
      overall determination for the role binding.

  Fields:
    membership: Indicates whether the principal is listed as a denied
      principal in the deny rule, either directly or through membership in a
      principal set.
    relevance: The relevance of the principal's status to the overall
      determination for the role binding.
  """

    class MembershipValueValuesEnum(_messages.Enum):
        """Indicates whether the principal is listed as a denied principal in the
    deny rule, either directly or through membership in a principal set.

    Values:
      MEMBERSHIP_MATCHING_STATE_UNSPECIFIED: Not specified.
      MEMBERSHIP_MATCHED: The principal in the request matches the principal
        in the policy. The principal can be included directly or indirectly: *
        A principal is included directly if that principal is listed in the
        role binding. * A principal is included indirectly if that principal
        is in a Google group, Google Workspace account, or Cloud Identity
        domain that is listed in the policy.
      MEMBERSHIP_NOT_MATCHED: The principal in the request doesn't match the
        principal in the policy.
      MEMBERSHIP_UNKNOWN_INFO: The principal in the policy is a group or
        domain, and the sender of the request doesn't have permission to view
        whether the principal in the request is a member of the group or
        domain.
      MEMBERSHIP_UNKNOWN_UNSUPPORTED: The principal is an unsupported type.
    """
        MEMBERSHIP_MATCHING_STATE_UNSPECIFIED = 0
        MEMBERSHIP_MATCHED = 1
        MEMBERSHIP_NOT_MATCHED = 2
        MEMBERSHIP_UNKNOWN_INFO = 3
        MEMBERSHIP_UNKNOWN_UNSUPPORTED = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the principal's status to the overall determination
    for the role binding.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Not specified.
      HEURISTIC_RELEVANCE_NORMAL: The data point has a limited effect on the
        result. Changing the data point is unlikely to affect the overall
        determination.
      HEURISTIC_RELEVANCE_HIGH: The data point has a strong effect on the
        result. Changing the data point is likely to affect the overall
        determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        HEURISTIC_RELEVANCE_NORMAL = 1
        HEURISTIC_RELEVANCE_HIGH = 2
    membership = _messages.EnumField('MembershipValueValuesEnum', 1)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 2)