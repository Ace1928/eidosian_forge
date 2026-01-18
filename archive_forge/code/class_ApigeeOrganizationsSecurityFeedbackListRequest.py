from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityFeedbackListRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityFeedbackListRequest object.

  Fields:
    filter: Optional. Allow filtering.
    pageSize: Optional. The maximum number of feedback reports to return. The
      service may return fewer than this value.
      LINT.IfChange(documented_page_size_limits) If unspecified, at most 100
      feedback reports will be returned. The maximum value is 1000; values
      above 1000 will be coerced to 1000. LINT.ThenChange(//depot/google3/edge
      /sense/boq/service/v1/securityfeedback/securityfeedback_rpc.go:page_size
      _limits)
    pageToken: Optional. A page token, received from a previous
      `ListSecurityFeedback` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      `ListSecurityFeedback` must match the call that provided the page token.
    parent: Required. Name of the organization. Format: `organizations/{org}`.
      Example: organizations/apigee-organization-name/securityFeedback
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)