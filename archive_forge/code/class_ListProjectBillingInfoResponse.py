from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListProjectBillingInfoResponse(_messages.Message):
    """Request message for `ListProjectBillingInfoResponse`.

  Fields:
    nextPageToken: A token to retrieve the next page of results. To retrieve
      the next page, call `ListProjectBillingInfo` again with the `page_token`
      field set to this value. This field is empty if there are no more
      results to retrieve.
    projectBillingInfo: A list of `ProjectBillingInfo` resources representing
      the projects associated with the billing account.
  """
    nextPageToken = _messages.StringField(1)
    projectBillingInfo = _messages.MessageField('ProjectBillingInfo', 2, repeated=True)