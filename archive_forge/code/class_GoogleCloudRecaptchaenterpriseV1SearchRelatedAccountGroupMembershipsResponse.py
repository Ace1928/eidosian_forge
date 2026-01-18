from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1SearchRelatedAccountGroupMembershipsResponse(_messages.Message):
    """The response to a `SearchRelatedAccountGroupMemberships` call.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    relatedAccountGroupMemberships: The queried memberships.
  """
    nextPageToken = _messages.StringField(1)
    relatedAccountGroupMemberships = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1RelatedAccountGroupMembership', 2, repeated=True)