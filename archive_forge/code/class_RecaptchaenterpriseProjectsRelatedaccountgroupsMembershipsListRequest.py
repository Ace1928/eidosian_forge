from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsRelatedaccountgroupsMembershipsListRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsRelatedaccountgroupsMembershipsListRequest
  object.

  Fields:
    pageSize: Optional. The maximum number of accounts to return. The service
      might return fewer than this value. If unspecified, at most 50 accounts
      are returned. The maximum value is 1000; values above 1000 are coerced
      to 1000.
    pageToken: Optional. A page token, received from a previous
      `ListRelatedAccountGroupMemberships` call. When paginating, all other
      parameters provided to `ListRelatedAccountGroupMemberships` must match
      the call that provided the page token.
    parent: Required. The resource name for the related account group in the
      format `projects/{project}/relatedaccountgroups/{relatedaccountgroup}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)