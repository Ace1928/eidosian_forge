from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsRelatedaccountgroupsListRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsRelatedaccountgroupsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of groups to return. The service
      might return fewer than this value. If unspecified, at most 50 groups
      are returned. The maximum value is 1000; values above 1000 are coerced
      to 1000.
    pageToken: Optional. A page token, received from a previous
      `ListRelatedAccountGroups` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      `ListRelatedAccountGroups` must match the call that provided the page
      token.
    parent: Required. The name of the project to list related account groups
      from, in the format `projects/{project}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)