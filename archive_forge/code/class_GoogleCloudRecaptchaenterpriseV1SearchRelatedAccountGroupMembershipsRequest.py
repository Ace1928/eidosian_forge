from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsRequest(_messages.Message):
    """The request message to search related account group memberships.

  Fields:
    accountId: Optional. The unique stable account identifier used to search
      connections. The identifier should correspond to an `account_id`
      provided in a previous `CreateAssessment` or `AnnotateAssessment` call.
      Either hashed_account_id or account_id must be set, but not both.
    hashedAccountId: Optional. Deprecated: use `account_id` instead. The
      unique stable hashed account identifier used to search connections. The
      identifier should correspond to a `hashed_account_id` provided in a
      previous `CreateAssessment` or `AnnotateAssessment` call. Either
      hashed_account_id or account_id must be set, but not both.
    pageSize: Optional. The maximum number of groups to return. The service
      might return fewer than this value. If unspecified, at most 50 groups
      are returned. The maximum value is 1000; values above 1000 are coerced
      to 1000.
    pageToken: Optional. A page token, received from a previous
      `SearchRelatedAccountGroupMemberships` call. Provide this to retrieve
      the subsequent page. When paginating, all other parameters provided to
      `SearchRelatedAccountGroupMemberships` must match the call that provided
      the page token.
  """
    accountId = _messages.StringField(1)
    hashedAccountId = _messages.BytesField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)