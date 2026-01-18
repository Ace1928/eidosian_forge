from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssertionClaimsBehaviorValueValuesEnum(_messages.Enum):
    """Required. The behavior for how OIDC Claims are included in the
    `assertion` object used for attribute mapping and attribute condition.

    Values:
      ASSERTION_CLAIMS_BEHAVIOR_UNSPECIFIED: No assertion claims behavior
        specified.
      MERGE_USER_INFO_OVER_ID_TOKEN_CLAIMS: Merge the UserInfo Endpoint Claims
        with ID Token Claims, preferring UserInfo Claim Values for the same
        Claim Name. This option is available only for the Authorization Code
        Flow.
      ONLY_ID_TOKEN_CLAIMS: Only include ID Token Claims.
    """
    ASSERTION_CLAIMS_BEHAVIOR_UNSPECIFIED = 0
    MERGE_USER_INFO_OVER_ID_TOKEN_CLAIMS = 1
    ONLY_ID_TOKEN_CLAIMS = 2