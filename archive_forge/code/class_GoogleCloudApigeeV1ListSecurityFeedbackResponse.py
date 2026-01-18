from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSecurityFeedbackResponse(_messages.Message):
    """Response for ListSecurityFeedback

  Fields:
    nextPageToken: A token that can be sent as `page_token` in
      `ListSecurityFeedbackRequest` to retrieve the next page. If this field
      is omitted, there are no subsequent pages.
    securityFeedback: List of SecurityFeedback reports.
  """
    nextPageToken = _messages.StringField(1)
    securityFeedback = _messages.MessageField('GoogleCloudApigeeV1SecurityFeedback', 2, repeated=True)